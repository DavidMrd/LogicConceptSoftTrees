"""
Train Resnet Network using the MonuMAI-200-2011 dataset
"""
# from cProfile import label
# from lib2to3.pytree import Base
# import pdb
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ltn
import math
import torch
from torch.utils.data import DataLoader
from MonuMAI.analysis import Logger, AverageMeter, accuracy, binary_accuracy
from torchvision.models import resnet50, ResNet50_Weights
from MonuMAI.Monumai import get_dataset, find_class_imbalance
from MonuMAI.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from MonuMAI.template_model import SDT
from MonuMAI.symbolic_utils import compute_accuracy,compute_sat_level,get_axioms_attr,get_axioms_attr_joint,get_axioms_classes, SatAgg, LogitsToPredicateResnetTree, LogitsToPredicateMulticlass,LogitsToPredicateMultilabel

def run_epoch(P_attr,P_classifier,P_full, optimizer, loader, total_loss_meter,ccentropy_loss_meter, sym_loss_metter, acc_meter, sat_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        for p in [P_attr,P_classifier,P_full]:
            p.train()
    else:
        for p in [P_attr,P_classifier,P_full]:
            p.eval()   
    resnet = P_full.model.first_model
    sdt = P_full.model.second_model
    for batch_idx, data in enumerate(loader):           
          
        if attr_criterion is None:
            inputs, labels = data
        else: 
            if args.n_attributes <= 1:
                raise BaseException("only 1 attribute is not allowed")
            inputs, labels, attr_labels = data
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var           
       

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        P_full.model.reset_penalty()
        optimizer.zero_grad()        
       
        ######################## add labels knowledge       
        #axioms_classes = get_axioms_classes(inputs_var,labels,is_training, predicate)       
        # add attributes knowledge              
        #axioms_attr = get_axioms_attr(inputs_var,attr_labels,is_training,predicate)           
        axioms_attr_joint = get_axioms_attr_joint(inputs_var, is_training,P_attr,P_full)    
        ############################################
        losses = []
        axioms = axioms_attr_joint
        sat_agg = SatAgg(*axioms)
        sat_meter.update(sat_agg,inputs.size(0))

        penalty_sym =  P_full.model.get_penalty()  if is_training else 0.0
        attr_outputs = resnet.forward(inputs_var)  
        tree_outputs      = sdt.forward(attr_outputs,is_training)
        if is_training:
            outputs, penalty = tree_outputs
        else:
            outputs = tree_outputs
            penalty =  0.0

        loss_classic = criterion(outputs, labels_var)  + penalty
        loss_symbolic = (1. - sat_agg)*2 + penalty_sym
        sym_loss_metter.update(loss_symbolic)
        ccentropy_loss_meter.update(loss_classic)
        loss_main = loss_classic + loss_symbolic
        losses.append(loss_main)
        
          
        acc = accuracy(outputs, labels, topk=(1,)) #only care about class prediction accuracy
        acc_meter.update(acc[0], inputs.size(0))
            
        if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
            for i in range(len(attr_criterion)):
                losses.append(args.attr_loss_weight * attr_criterion[i](attr_outputs[:,i].type(torch.cuda.FloatTensor), attr_labels_var[:, i]))             
        

        if attr_criterion is not None:
            #cotraining, loss by class prediction and loss by attribute prediction have the same weight
            total_loss_attr = sum(losses[1:])
            if args.normalize_loss:
                total_loss_attr = total_loss_attr / (1 + args.attr_loss_weight * args.n_attributes)
            
            total_loss = losses[0] + total_loss_attr
            #print("total loss attr: "+ str(total_loss_attr))
        else: #finetune
            #raise NotImplementedError("Not implemented")
            total_loss = sum(losses)
        total_loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:           
            total_loss.backward()
            optimizer.step()
   
    return total_loss_meter, ccentropy_loss_meter, sym_loss_metter, sat_meter, acc_meter

def train(resnet,sdt, args):   

    print("Start training")
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        #raise NotImplementedError("Not implemented")
        train_data_path = 'MonuMAI/csv/train_onehot_seed'+str(args.seed)+'.csv'#os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    
    P_attr = ltn.Predicate(LogitsToPredicateMultilabel(resnet))
    P_classifier = ltn.Predicate(LogitsToPredicateMulticlass(sdt))
   
    P_full = ltn.Predicate(LogitsToPredicateResnetTree(resnet, sdt))

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            #raise NotImplementedError("Not implemented")
            assert(imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()if torch.cuda.is_available() else torch.FloatTensor([ratio])))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None
        print("WARNING: Not using attr!")

    optimizer = torch.optim.Adam(P_full.model.parameters(),
                                 lr= args.lr ,
                                 weight_decay=5e-4 )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)


    train_dataset, val_dataset, test_dataset  = get_dataset(train_file='MonuMAI/csv/train_onehot_seed'+str(args.seed)+'.csv', val_file='MonuMAI/csv/val_onehot_seed'+str(args.seed)+'.csv', test_file = 'MonuMAI/csv/test_onehot_seed'+str(args.seed)+'.csv' , use_attr = args.use_attr, no_img = args.no_img, use_inception = False)

    # if args.ckpt: #retraining
    #     raise NotImplementedError("args.ckpt Not implemented, what is this!!!??")
           
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_total_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_sat_meter = AverageMeter()
        train_ccentropy_loss_meter = AverageMeter()
        train_sym_loss_metter = AverageMeter()
        if args.no_img:
            raise NotImplementedError("no_img Not implemented")
        else:
            train_total_loss_meter,train_ccentropy_loss_meter, train_sym_loss_metter, train_sat_meter, train_acc_meter = run_epoch(P_attr,P_classifier,P_full, optimizer, train_loader, train_total_loss_meter, train_ccentropy_loss_meter, train_sym_loss_metter, train_acc_meter, train_sat_meter, criterion, attr_criterion, args, is_training=True)
 
       # evaluate on val set
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()        
        val_sat_meter = AverageMeter()
        val_ccentropy_loss_meter = AverageMeter()
        val_sym_loss_metter = AverageMeter()
        with torch.no_grad():
            val_loss_meter,val_ccentropy_loss_meter, val_sym_loss_metter,val_sat_meter, val_acc_meter = run_epoch(P_attr,P_classifier,P_full, optimizer, val_loader, val_loss_meter, val_ccentropy_loss_meter, val_sym_loss_metter, val_acc_meter, val_sat_meter, criterion, attr_criterion, args, is_training=False)

       

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(P_full.model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))

        train_loss_avg = train_total_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain cee_loss: %.4f\tTrain sym_loss: %.4f\tTrain sat: %.4f\t'
                'Val loss: %.4f\tVal cce_loss: %.4f\tVal sym_loss: %.4f\tVal sat: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg,train_ccentropy_loss_meter.avg, train_sym_loss_metter.avg, train_sat_meter.avg, val_loss_avg, val_ccentropy_loss_meter.avg, val_sym_loss_metter.avg, val_sat_meter.avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())

        if epoch % args.save_step == 0:
            torch.save(P_full.model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 200:
            print("Early stopping because acc hasn't improved for a long time")
            break


    
def train_X_to_C_to_y_soft_decision_tree_symbolic_fusion(args):  
    
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, args.n_attributes)

    sdt = SDT(input_dim=args.n_attributes,output_dim=N_CLASSES,depth=4,use_cuda=torch.cuda.is_available())
    resnet = resnet.cuda() if torch.cuda.is_available() else resnet
    sdt = sdt.cuda() if torch.cuda.is_available() else sdt 
    
    train(resnet,sdt,args)
    

def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='MonuMAI Training')
    parser.add_argument('exp', type=str,
                        choices=[ 'Joint_SoftDecisionTreeSymbolicFusion'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')  
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-uncertain_labels', action='store_true',
                        help='whether to use (normalized) attribute certainties as labels')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-expand_dim', type=int, default=0,
                        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-connect_CY', action='store_true',
                        help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    return (args,)
