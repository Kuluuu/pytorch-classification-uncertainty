import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
import logging


def train_model(
    model,
    dataloaders,
    num_classes,
    acc_criterion,
    uncertainty_criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    use_uncertainty=False,
):
    timestamp = "{}".format(str(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
    postfix = timestamp + '_' + str(num_epochs) + 'epoch'
    log_save_dir = "./log/" + postfix + ".log"
    logging.basicConfig(level=logging.INFO,
                    filename=log_save_dir,
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"acc_loss": [], "uncertainty_loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                logging.info("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                logging.info("Validating...")
                model.eval()  # Set model to evaluate mode

            running_acc_loss = 0.0
            running_uncertainty_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if use_uncertainty:
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs, uncertainty = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        acc_loss = acc_criterion(outputs, labels)
                        uncertainty_loss = uncertainty_criterion(
                            uncertainty, y.float(), epoch, num_classes, 10, device
                        )
                        loss = acc_loss + uncertainty_loss

                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(uncertainty)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        print("ERROR!!! - always Uncertainty")

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_acc_loss += acc_loss.item() * inputs.size(0)
                running_uncertainty_loss += uncertainty_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_acc_loss = running_acc_loss / len(dataloaders[phase].dataset)
            epoch_uncertainty_loss = running_uncertainty_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["acc_loss"].append(epoch_acc_loss)
            losses["uncertainty_loss"].append(epoch_uncertainty_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(
                "{} acc_loss: {:.4f} acc_loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_acc_loss, epoch_uncertainty_loss, epoch_acc
                )
            )
            logging.info("{} acc_loss: {:.4f} acc_loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_acc_loss, epoch_uncertainty_loss, epoch_acc
                ))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
