import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from sklearn.metrics import f1_score
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from score_functions import prediction_scores, Scorer
from train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        post_epoch_fn: Callable = None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        epochs_without_improvement = 0
        train_loss, train_score, test_loss, test_score = [], [], [], []
        best_acc = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.

            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            actual_num_epochs += 1

            train_result = self.train_epoch(dl_train, **kw)
            train_loss += train_result.losses
            train_score.append(train_result.score)

            test_result = self.test_epoch(dl_test, **kw)
            test_loss += test_result.losses
            test_score.append(test_result.score)

            # Early stopping and checkpoint
            if best_acc is None or test_result.score > best_acc:
                best_acc = test_result.score
                epochs_without_improvement = 0
                if checkpoints:
                    filename = os.getcwd()+'/'+checkpoints+'.pth'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    self.save_checkpoint(filename)
            else:
                epochs_without_improvement += 1
            if early_stopping and epochs_without_improvement > early_stopping:
                break

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_score, test_loss, test_score)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save({"model_state": self.model.state_dict()}, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        score_fn: Scorer,
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        TP, FP, TN, FN = 0, 0, 0, 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                TP += batch_res.TP
                FP += batch_res.FP
                TN += batch_res.TN
                FN += batch_res.FN

            avg_loss = sum(losses) / num_batches
            score = score_fn(TP, FP, TN, FN)
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss: {avg_loss:.3f}, "
                f"Score {score_fn.name}:  {score:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, score=score)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, true_threshold=0.5):
        super().__init__(model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.true_threshold = true_threshold

    def train_batch(self, batch) -> BatchResult:
        x, y = batch

        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate score function
        output = self.model(x).squeeze(1)
        self.optimizer.zero_grad()
        loss = self.loss_fn(output, y)

        loss.backward()
        self.optimizer.step()

        predictions = torch.sigmoid(output)
        predictions = (predictions > self.true_threshold).int()

        return BatchResult(loss.item(), *prediction_scores(predictions, y))

    def test_batch(self, batch) -> BatchResult:
        x, y = batch

        #  Evaluate the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Calculate score function
        output = self.model(x).squeeze(1)
        loss = self.loss_fn(output, y)

        predictions = torch.sigmoid(output)
        predictions = (predictions > self.true_threshold).int()

        return BatchResult(loss.item(), *prediction_scores(predictions, y))

    def test_batch_niv(self, batch) -> BatchResult:
        sequences, label = batch
        seq_len = y.shape[1]

        with torch.no_grad():
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            output, hidden_state = self.model(x, self.test_hidden_state)
            self.test_hidden_state = hidden_state
            loss = 0
            num_correct = 0
            for i, seq in enumerate(sequences):
                logits = output[:, i, :].squeeze(1)
                pred = torch.argmax(logits, dim=1)
                loss += self.loss_fn(logits, y[:, i])
                num_correct += (y[:, i] == pred).sum()
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)
