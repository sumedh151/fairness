import torch
import torch.optim as optim
import torch.nn as nn

def calculate_gradient(model):
    """
    Calculate the L2 norm of the gradients for a given model.

    Args:
    - model (torch.nn.Module): The model whose gradients will be calculated.

    Returns:
    - float: The L2 norm of the gradients.
    """
    batch_gradient_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            batch_gradient_norm += p.grad.data.norm(2).item() ** 2
    batch_gradient_norm = batch_gradient_norm ** 0.5
    return batch_gradient_norm

def train_model(encoder, predictor, adversaries, num_epochs, num_non_sensitive , num_sensitive, 
                train_dataloader, criterion_enc, criterion_pred, criterion_adversaries, 
                optimizer_enc, optimizer_pred, optimizer_adversaries, alpha=10, max_norm=1.0):

    num_epochs = num_epochs

    encoder.train()
    predictor.train()
    for adversary in adversaries:
        adversary.train()

    gradient_norms_enc = []
    gradient_norms_pred = []
    gradient_norms_adversaries = [[] for _ in range(len(adversaries))]

    loss_adversaries = [[] for _ in range(len(adversaries))]
    loss_enc = []
    loss_pred= []
    loss_comb= []

    epochs_total = []


    for epoch in range(num_epochs):
        
        total_gradient_norm_adversaries = [0.0 for _ in range(len(adversaries))]
        total_gradient_norm_enc = 0.0
        total_gradient_norm_pred = 0.0
        num_batches_adversaries = 0
        num_batches_other = 0

        for data, labels in train_dataloader: 

            for adversary in adversaries:
                adversary.zero_grad()
            
            x_recon = encoder(data)
            adv_pred = [adversary(x_recon) for adversary in adversaries]
            la_adversaries = [criterion_adversaries[i](adv_pred[i], data[:,num_non_sensitive + i].view(-1,1).float()) for i in range(len(criterion_adversaries))]


            for i in range(len(la_adversaries)):
                if i != len(la_adversaries)-1:
                    la_adversaries[i].backward(retain_graph = True)
                else:
                    la_adversaries[i].backward()


            for i in range(len(adversaries)):
                total_gradient_norm_adversaries[i] += calculate_gradient(adversaries[i])
            
            num_batches_adversaries += 1

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            
            for i in range(len(adversaries)):
                torch.nn.utils.clip_grad_norm_(adversaries[i].parameters(), 1.0)

            for i in range(len(adversaries)):
                optimizer_adversaries[i].step()
            
        for data, labels in train_dataloader: 
            pass
        
        encoder.zero_grad()
        predictor.zero_grad()
        x_recon = encoder(data)
        y_pred = predictor(x_recon)
        adv_pred = [adversary(x_recon) for adversary in adversaries]

        lx = criterion_enc(x_recon, data[:,:4])
        lp = criterion_pred(y_pred, labels)
        la_adversaries = [criterion_adversaries[i](adv_pred[i], data[:,num_non_sensitive + i].view(-1,1).float()) for i in range(len(criterion_adversaries))]

        combined_loss = lx + lp - (10 * sum(la_adversaries))
        combined_loss.backward()

        for i in range(len(adversaries)):
            total_gradient_norm_adversaries[i] += calculate_gradient(adversaries[i])
            
        total_gradient_norm_enc += calculate_gradient(encoder)

        total_gradient_norm_pred += calculate_gradient(predictor)

        num_batches_other += 1
        num_batches_adversaries += 1

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        for i in range(len(adversaries)):
            torch.nn.utils.clip_grad_norm_(adversaries[i].parameters(), 1.0)

        optimizer_pred.step()
        optimizer_enc.step()
        
        average_gradient_norm_adversaries = list(map(lambda x: x/num_batches_adversaries,  total_gradient_norm_adversaries))

        for i in range(len(adversaries)):
            gradient_norms_adversaries[i].append(average_gradient_norm_adversaries[i])
        
        average_gradient_norm_enc = total_gradient_norm_enc / num_batches_other
        gradient_norms_enc.append(average_gradient_norm_enc)

        average_gradient_norm_pred = total_gradient_norm_pred / num_batches_other
        gradient_norms_pred.append(average_gradient_norm_pred)
        
        for i in range(len(adversaries)):
            loss_adversaries[i].append(la_adversaries[i].item())
        loss_enc.append(lx.item())
        loss_pred.append(lp.item())
        loss_comb.append(combined_loss.item())

        epochs_total.append(epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], ' + ', '.join(f'loss adversary {i} : {la_adversaries[i].item():.4f}' for i in range(len(loss_adversaries))) + f', Loss Recon: {lx.item():.4f}, Loss P: {lp.item():.4f}, Loss Comb: {combined_loss.item():.4f}')
    
    return encoder, predictor, adversaries, epochs_total, gradient_norms_enc, gradient_norms_pred, gradient_norms_adversaries, loss_enc, loss_pred, loss_adversaries, loss_comb