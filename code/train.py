from model_base import *
from model_mixture_of_exprts import *
from model_product_of_exprts import *
from model_general_gmoe_prob import *
from model_general_gmoe_main import *
from helpers import *

# track the training
from tensorboardX import SummaryWriter


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


def count_parameters(model):
    '''
    This function calculates trainable parameters.
    :param model:
    :return:
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainer(args):

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data loaders
    train_x, train_y, test_x, test_y = call_datasets(train_noise=args.train_noise, test_noise=args.test_noise)
    train_loader, test_loader = get_dataloaders(train_x, train_y, test_x, test_y, args.batch)
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)

    # networks and optimizer:
    if args.net == 'moe':
        network = UNetMoE(args.width, args.dilation).cuda()

    elif args.net == 'poe':
        network = UNetPoE(args.width, args.dilation).cuda()

    elif args.net == 'gmoe':

        if args.gate_type == 1:
            standard_gate = False
            random_gate = False
            avg_gate = True
        elif args.gate_type == 2:
            standard_gate = False
            random_gate = True
            avg_gate = False
        else:
            standard_gate = True
            random_gate = False
            avg_gate = False

        network = Generalised_GMoE(width=args.width,
                                   ks=args.ks,
                                   mu=0,
                                   sigma=1,
                                   dilation=1,
                                   num_elayers=args.num_elayers,
                                   standard_gate=standard_gate,
                                   random_gate=random_gate,
                                   avg_gate=avg_gate).cuda()

    elif args.net == 'pgmoe':
        network = Generalised_GMoE_VI(width=args.width,
                                      num_elayers=(args.no_experts,
                                                    args.no_experts,
                                                    args.no_experts,
                                                    args.no_experts,
                                                    args.no_experts)
                                      ).cuda()
    else:
        network = UNet(args.width, args.dilation).cuda()

    parameters = count_parameters(network)
    print('Model params: %d' % parameters)

    model_config = 'e' + str(args.epochs) + '_' + args.net + '_l' + str(args.lr) + '_d' + str(args.dilation) + '_w' + str(args.width) + '_tr' + str(args.train_noise) + '_te' + str(args.test_noise) + '_p' + str(parameters) + '_ep' + str(args.epsilon)

    # Log:
    writer = SummaryWriter('../logs/' + model_config)

    if args.loss_fun == 'dice':
        criterion = SoftDiceLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.AdamW(network.parameters(), lr=args.lr)
    steps_each_epoch = 60000 // args.batch
    total_steps = steps_each_epoch*args.epochs
    best_val_acc = 0.0

    for j in range(total_steps):

        network.train()

        try:
            images, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            test_iterator = iter(test_loader)
            images, labels = next(train_iterator)

        optimizer.zero_grad()

        if args.net == 'pgmoe':
            outputs, qy = network(images)
            log_qy = torch.log(qy+1e-20)
            g = nn.Parameter(torch.log(torch.Tensor([1.0/args.no_experts])).cuda())
            KLD = torch.sum(qy*(log_qy - g), dim=-1).mean()
            if args.loss_fun == 'dice':
                loss = criterion(torch.sigmoid(outputs / args.temp), labels) + 0.1*KLD
            else:
                loss = criterion(torch.sigmoid(outputs / args.temp), labels) + 0.1*KLD
        else:
            outputs = network(images)
            if args.loss_fun == 'dice':
                loss = criterion(torch.sigmoid(outputs / args.temp), labels)
            else:
                loss = criterion(torch.sigmoid(outputs / args.temp), labels)

        loss.backward()
        optimizer.step()

        if args.lr_decay == 1:
            optimizer.param_groups[0]['lr'] = args.lr * (1 - j / total_steps) ** 0.99
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = args.lr

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct = preds.eq(labels.view_as(preds)).sum().item()
        train_acc = correct / images.size()[0] / images.size()[1] / images.size()[2] / images.size()[3]
        running_loss = loss.item()
        train_acc = 100 * train_acc

        if j % steps_each_epoch == 0 and j != 0:
            network.eval()
            val_acc = 0
            counter_v = 0
            for (v_img, v_target) in test_iterator:
                counter_v += 1
                if args.epsilon > 0:
                    v_img.requires_grad = True
                    if args.net == 'pgmoe':
                        v_output, qy = network(v_img)
                    else:
                        v_output = network(v_img)
                    v_loss = criterion(torch.sigmoid(v_output), v_target)
                    network.zero_grad()
                    v_loss.backward()
                    data_grad = v_img.grad.data
                    perturbed_data = fgsm_attack(v_img, args.epsilon, data_grad)
                    v_output = network(perturbed_data)
                else:
                    v_output = network(v_img)
                v_pred = (torch.sigmoid(v_output) > 0.5).float()  # get the index of the max log-probability
                v_correct = v_pred.eq(v_target.view_as(v_pred)).sum().item()
                val_acc += v_correct / v_img.size()[0] / v_img.size()[1] / v_img.size()[2] / v_img.size()[3]
            val_acc = 100 * val_acc / counter_v
            print('[step %d] loss: %.4f, lr: %.4f, train acc:%.4f, test acc: %.4f' % (j + 1, running_loss, current_lr, train_acc, val_acc))
            if val_acc > best_val_acc:
                # save model
                torch.save(network.state_dict(), model_config + '_model.pt')
                # update best val acc
                best_val_acc = max(best_val_acc, val_acc)
            # log in:
            writer.add_scalar('loss', loss.item(), j)
            writer.add_scalar('val acc', val_acc, j)

        log_gradients_in_model(network, writer, j)

    print('Finished Training\n')

    torch.save(network.state_dict(), model_config + '_lastmodel.pt')

    if args.net == 'moe':
        bestnetwork = UNetMoE(args.width, args.dilation)
        lastnetwork = UNetMoE(args.width, args.dilation)
    elif args.net == 'poe':
        bestnetwork = UNetPoE(args.width, args.dilation).cuda()
        lastnetwork = UNetPoE(args.width, args.dilation)
    else:
        bestnetwork = UNet(args.width, args.dilation)
        lastnetwork = UNet(args.width, args.dilation)

    bestnetwork.load_state_dict(torch.load(model_config + '_model.pt'))
    bestnetwork.eval()

    lastnetwork.load_state_dict(torch.load(model_config + '_lastmodel.pt'))
    lastnetwork.eval()

    return bestnetwork, lastnetwork

    # # testing:
    # predictions = []
    # test_x_o = np.shape(test_x)[0]
    # for i in range(test_x_o):
    #     data = np.expand_dims(test_x[i, :, :, :], axis=0)
    #     data = torch.from_numpy(data).type(torch.FloatTensor).to('cuda')
    #     pred = bestnetwork(data)
    #     pred = (pred > 0.5).float()
    #     predictions += list(pred.data.cpu().numpy())
    #
    # print('Testing is done')

