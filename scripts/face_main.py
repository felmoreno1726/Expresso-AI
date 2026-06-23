import os, time, argparse, datetime, math
import pandas as pd
from tqdm import tqdm
from warnings import warn

import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import environment_config as cfg
from expresso_ai.models import initialize_model
from expresso_ai.data import make_dataloaders
from expresso_ai.data.FaceDataset import make_datasets
from expresso_ai.interpret.ModelInterpreter import ModelInterpreter
from expresso_ai.metrics.MetricsWrapper import MetricsWrapper
from expresso_ai.utils.generate_metadata import get_training_indices, generate_metadata, UnbalanceableDataError
from expresso_ai.utils.AverageMeter import AverageMeter
from expresso_ai.utils.logger import Logger
from expresso_ai.utils.parser_helpers import str2bool
from expresso_ai.utils.model_helpers import robust_load_weights, load_checkpoint, save_checkpoint
from expresso_ai.optim.freezer import Freezer, TwoStepFreezer

print('Torch version: ', torch.__version__)
print('Torch\'s respective cuda verson: ', torch.version.cuda)
#Machine specific environment variables 
batch_size = cfg.BATCH_SIZE
num_workers = cfg.NUM_WORKERS
pin_memory = cfg.NVIDIA_GPU
parallelize_gpus = cfg.PARALLELIZE_GPUS
default_device=cfg.DEFAULT_DEVICE
enable_apex = cfg.ENABLE_APEX

if enable_apex: 
    from apex import amp
'''
Train/test entry point for Expresso-AI facial-video models
'''
#Filestructure constants
file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.join(os.path.dirname(file_path), os.pardir)
PRETRAINED_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'pre-trained-weights')
CHECKPOINTS_PATH = os.path.join(ROOT_DIR, 'weights')
LOGGER_PATH = os.path.join(ROOT_DIR, 'experiments_log.csv')
TENSORBOARD_PATH = os.path.join(ROOT_DIR, 'tensorboard_logs')
STATIC_PATH = os.path.join(ROOT_DIR, 'static')
INTERPRETATION_PATH = os.path.join(ROOT_DIR, 'interpretation')
AUS_DATA_PATH = os.path.join(ROOT_DIR, 'AUs')
INTERPRETATION_LOGGER_PATH = os.path.join(INTERPRETATION_PATH, 'interpretation_log.csv')
#constant parameters


def main():
    global args, best_validation_metric_score, weight_decay, momentum, logger, writer
    #model definition
    #criterion = nn.BCEWithLogitsLoss().to(device=default_device)
    criterion = nn.MSELoss().to(device=default_device)
    model, img_size, img_channels = initialize_model(**vars(args), pretrained_weights_path=PRETRAINED_WEIGHTS_PATH)
    model = model.to(device=default_device)
    #log model to tensorboard
    #writer.add_graph(model)
    #Create dataloaders
    training_indices = get_training_indices(args.data_path, args.val_indices, args.test_indices)
    print("training indices: ", training_indices)
    try:
        metadata = generate_metadata(args.window_size, args.overlap, args.input_dilation, args.data_path, img_size, training_indices, args.val_indices, args.test_indices)
    except UnbalanceableDataError:
        warn("Unbalanceable data error")
        #This exception is thrown whenever any train, test or validation metadata contain no + or no - samples.
        return
    #fix image size
    datasets = make_datasets(args.data_path, metadata, static_path=STATIC_PATH, img_size=img_size)
    dataloaders = make_dataloaders(datasets, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    # Initialize metrics
    metrics = MetricsWrapper(args.metric, args.additional_metrics)
    #model definition special parameters
    if enable_apex:
        print('using Nvidia Apex')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if torch.cuda.device_count() > 1 and parallelize_gpus:
        print("Using {} gpus.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    #benchmark model (performance boost)
    cudnn.benchmark = True
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    epoch = 0
    if args.load is not None:
        checkpoint = load_checkpoint(args.load, args.load_best)
        if checkpoint is not None:
            #get the metric used for validation
            metric = checkpoint["metric"]
            print('checkpoint: ', checkpoint.keys())
            print('Loading checkpoint for epoch %05d with metric %s and score %.5f ' % (checkpoint['epoch'], metric, checkpoint['best_validation_metric_score']))
            state = checkpoint['state_dict']
            robust_load_weights(model, state)
            epoch = checkpoint['epoch']
            best_validation_metric_score = checkpoint['best_validation_metric_score']
            print('loaded model at epoch {} with best_validation_metric_score {}'.format(epoch, best_validation_metric_score))
        else:
            raise Exception('Error: could not read checkpoint at {}.'.format(args.load))
    #Run training routine
    if args.train:
        #Initialize optimization parameters
        # Load optimizer to the epoch the model loaded is at.
        #for epoch in range(0, epoch):
            #adjust_learning_rate(optimizer, epoch)
        optimization_params = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        if args.model_name.find("facenet_") != -1:
            #model is a partial transfer learning, defreeze everything from the start but the pre-trained portion
            try:
                layers_to_defreeze = len(list(model.modules())) - len(list(model.stem.modules()))
                #layers_to_defreeze = len(list(model.children())) - 1
            except torch.nn.modules.module.ModuleAttributeError:
                layers_to_defreeze = len(list(model.module.modules())) - len(list(model.module.stem.modules()))
                #layers_to_defreeze = len(list(model.module.modules())) - 1 
        else:
            #otherwise defreeze the last layer only
            layers_to_defreeze = 1
        #freezer = Freezer(model, defreeze_last_n = layers_to_defreeze)
        freezer = TwoStepFreezer(model, initial_step=1, step_size=1, defreeze_last_n=layers_to_defreeze)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **optimization_params)
        freezer.set_optimizer(optimizer)
        step_size = len(dataloaders["train"])
        print("Train dataloader 'step' size: ", step_size)
        #scheduler = torch.optim.lr_scheduler.StepLR(
                #optimizer,
                #step_size=10,
                #gamma=0.1,
                #last_epoch = -1 if epoch == 0 else epoch,
                #verbose=True
        #)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, lr, max_lr, 
                #step_size_up= math.floor(step_size / 2),
                #step_size_down= math.ceil(step_size / 2),
                #mode=  'triangular2', #choose from: triangular, triangular2, exp_range
                ##last_epoch= epoch * step_size,
                #verbose=True
        #)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = step_size,
                T_mult = 2,
                eta_min = min_lr,
                last_epoch = -1 if epoch == 0 else epoch,
                verbose = True
        )
        #Train routine
        #patience stops training if overfitting
        patience = 0
        epochs_to_train = epochs - epoch
        checkpoint = {}
        for epoch in tqdm(range(epoch, epochs), position=0, total=epochs_to_train):
            del checkpoint
            #adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            training_metrics = feed_data(dataloaders['train'],'train', model, criterion, epoch, metrics, optimizer=optimizer, scheduler=scheduler)
            #log training metrics
            log_entry = {"training_{}".format(metric_name): metric_score for metric_name, metric_score in training_metrics.items()}
            print('log entry: ', log_entry)
            logger.log(**log_entry)
            # evaluate on validation set
            validation_metrics = feed_data(dataloaders['val'], 'val', model, criterion, epoch, metrics)
            #log validation metrics
            log_entry={"validation_{}".format(metric_name): metric_score for metric_name, metric_score in validation_metrics.items()}
            logger.log(**log_entry)
            # remember best metric score and save checkpoint
            validation_metric_score = validation_metrics[args.metric]
            is_best = validation_metric_score < best_validation_metric_score
            print('is best: ', is_best)
            if is_best:
                best_validation_metric_score = validation_metric_score
                #log the changes
                log_entry = {"best_validation_{}".format(args.metric): best_validation_metric_score}
                logger.log(**log_entry)
                #reset patience
                patience = 0
            else:
                patience += 1
            #save checkpoint
            checkpoint = {
                'metric': args.metric,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_validation_metric_score': best_validation_metric_score,
            }
            save_checkpoint(checkpoint, is_best, filename=timestamp+".pth.tar", directory_path=CHECKPOINTS_PATH)#name of the weights saved will be the timestamp unique identifier
            print('checkpoint must have mutated with best: ', checkpoint.keys())
            #Optimizer
            #scheduler.step()
            freezer.step()
            #log to tensorboard writer is epoch is best
            writer.add_scalar("epoch/is_best", int(is_best), epoch)
            if patience >= args.patience_treshold:
                print(f"Model has not improved in {patience} epochs. Early stopping!")
                break
    #Run testing routine
    if args.test:
        #reload best state for validation. 
        print('loaded model for testing at epoch {} with best_validation_metric_score {}'.format(checkpoint["best"]["epoch"], checkpoint["best"]["best_validation_metric_score"]))
        robust_load_weights(model, checkpoint['best']['state_dict'])
        testing_metrics = feed_data(dataloaders['test'], 'test', model, criterion, epoch, metrics)
        log_entry = {'testing_{}'.format(metric_name): metric_score for metric_name, metric_score in testing_metrics.items()}
        logger.log(**log_entry)
        print('finished testing. args.interpret? ', args.interpret)
    #Run interpretation routine
    if args.interpret:
        print('Executing interpretation routine')
        robust_load_weights(model, checkpoint['best']['state_dict'])
        #use a logger to observe the model output predictions
        interpretation_metrics = feed_data(dataloaders['interpret'], 'interpret', model, criterion, epoch, metrics)
        log_entry = {'interpretation_{}'.format(metric_name): metric_score for metric_name, metric_score in interpretation_metrics.items()}
        logger.log(**log_entry)
        print('finished testing. args.interpret? ', args.interpret)


def feed_data(data_loader, phase, model, criterion, epoch, metrics, optimizer=None, scheduler=None):
    """feeds data to the model for one loop iteration of the given dataloader. Adjusts weights if phase is train
    phase (String): train, test, val or interpret.
    optimizer: optimizer method (SGD, Adam, etc) to optimizer loss function over. It is required for training phase only
    Returns the average metric of the model."""
    global writer, args
    if phase == 'train':
        print('Setting model to training mode.')
        model.train()
    else:
        #Set model in evaluation mode
        model.eval()
        print('Setting model to evaluation mode.')
        if phase == 'interpret':
            #initialize the model interpreter
            print('args.compute_attributions: ', args.compute_attributions)
            interpreter = ModelInterpreter(model, compute_attributions=args.compute_attributions, algorithm_name=args.interpretation_algorithm, visualization_methods=args.visualization_methods, signs=args.attribution_signs)
            #set baseline as interpreter's attribute
            #baseline_dimensions = (metadata['interpret']['window_size'], img_channels, *img_size)
            predictions_logger = pd.DataFrame()
    #time the operations
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #Reset hidden state of metrics
    metrics.reset()
    end = time.time()
    #load data
    for i, (sample_ids, participant_ids, window_index, imFace, labels) in tqdm(enumerate(data_loader), position=1, total=(len(data_loader))):
        # measure data loading time
        data_time.update(time.time() - end)
        if phase == 'interpret':
            torch.cuda.empty_cache()
            pass
        #to device
        imFace = imFace.to(device=default_device, non_blocking=True)
        labels = labels.to(device=default_device, non_blocking=True)
        this_batch_size = imFace.size(0)
        #set auto gradients
        autograd_flag = (phase == 'train' or phase == 'interpret')
        imFace.requires_grad_(autograd_flag)
        labels.requires_grad_(False)
        #optional faceGrid for some models
        inputs = (imFace)
        input_names = ["face"]
        # compute output
        if autograd_flag:
            if phase == 'train':
                output = model(inputs)
                #adjust labels with regularization noise
                labels = labels + torch.normal(mean=torch.ones(labels.shape)*args.noise_mean, std=torch.ones(labels.shape)*args.noise_std).to(device=default_device)
            elif phase == 'interpret':
                interpreter.interpretation_routine(inputs, input_names, labels.item(), 
                        sample_ids.item(), participant_ids.item(), window_index.item(),#data sample parameters
                        window_size=args.window_size, overlap=args.overlap, input_dilation=args.input_dilation,#dataloading constants
                        data_path=args.data_path, attributions_path=args.attributions_path, interpretation_path=INTERPRETATION_PATH, aus_data_path=AUS_DATA_PATH,#path parameters
                        generate_video_visualizations=args.generate_video_visualizations, generate_outstanding_frames_grid=args.generate_outstanding_frames_grid, generate_action_unit_cross_correlation=args.generate_action_unit_cross_correlation, generate_region_importances=args.generate_region_importances #parameters for routines to execute
                )
                if not args.interpretation_predict:
                    #skip inferencing and evaluation steps
                    continue
                output = model(inputs)
        else:
            with torch.no_grad():
                output = model(inputs)
        loss = criterion(output, labels)
        losses.update(loss.data.item(), this_batch_size)
        #calculate metric
        predictions = output.detach()
        metrics.update((predictions, labels), ids=participant_ids)
        if phase == 'train':
            # compute gradient and do SGD step
            optimizer.zero_grad()
            if enable_apex:
                with amp.scale_loss(loss, optimizer) as scalled_loss:
                    scalled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
        elif phase == 'interpret':
            #log prediction outputs
            interpretation_output = int(predictions.detach().cpu().numpy())
            predictions = {
                    'sample_id': sample_ids.item(),
                    'participant_id': participant_ids.item(),
                    'label': int(labels.item()),
                    'prediction': interpretation_output
            }
            predictions_logger = predictions_logger.append(predictions, ignore_index=True)
        #compute metrics
        metrics.compute_robust()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #print epoch results
        print('Epoch ({phase}): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               epoch, i, len(data_loader), phase=phase, 
               batch_time=batch_time,
               data_time=data_time, 
               loss=losses,
              ) + \
              metrics.report_metrics()
        )
        #Add metrics and loss to tensorboard log
        iteration = epoch * len(data_loader) + i if phase == 'train' or phase == 'val' else i
        writer.add_scalar(f"Loss/{phase}", losses.avg, iteration)
        for metric_name, metric_score in metrics.get_computed_metrics().items():
            writer.add_scalar(f"{metric_name}/{phase}", metric_score, iteration)
    #save predictions log before returning
    if phase == 'interpret':
        predictions_logger.to_csv(INTERPRETATION_LOGGER_PATH)
    return metrics.compute()


if __name__ == "__main__":
    #define command line parser
    parser = argparse.ArgumentParser(description='Main program train/test/interpret VideoFaceNets.')
    #define arguments to parse
    parser.add_argument('-d', '--data_path', nargs='?', default='./regression_dataset/', help="Default=\'./regression_dataset/\'. Path to processed dataset.")
    parser.add_argument('--attributions_path', nargs='?', default='./attributions/', help="Default=\'./attributions/\'. Path to were to read from or store attributions.")
    #Metadata generation parameters
    parser.add_argument('--window_size', type=int, default=300, help="Default=300. The number of frames in each data sample.")
    parser.add_argument('--overlap', type=float, default=0.5, help="Default=0.5. The percent of overlap between continuous data points in a video sample.")
    parser.add_argument('--input_dilation', type=int, default=1, help="Default=1. The dilation used on the input frames to the model.")
    parser.add_argument('--val_indices', nargs='+', default=[str(i) for i in range(200, 300)])
    parser.add_argument('--test_indices', nargs='+', default=[str(i) for i in range(200, 300)])
    #Routine parameters
    parser.add_argument('-l', '--load', type=str, nargs='?', default=None, help="Path to weigths to load into the model.")
    parser.add_argument('--load_best', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Loads the model at the best validation weights.")
    #type of routine(s) to run
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Run train model procedure.")
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Run the testing routine. If train flag is passed, it is executed after training")
    parser.add_argument('--interpret', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Run the validation procedure saving saliency maps to the interpretations directory")
    #training parameters
    parser.add_argument('--epochs', type=int, nargs='?', default=2, help="Default = 2. Sets the number of epochs to train for")
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0, help="Default = 0. Sets the weight decay regularization for the optimizer. ")
    parser.add_argument('--momentum', type=float, nargs='?', default=0, help="Default = 0. Sets the momentum of the SGD optimizer. ")
    parser.add_argument('--lr', type=float, nargs='?', default=1e-4, help="Default = 0. Sets the lr of the SGD optimizer. ")
    parser.add_argument('--dropout_prob', type=float, nargs='?', default=0, help="Default = 0. Sets the probability of dropout layer")
    parser.add_argument('--noise_mean', type=float, nargs='?', default=0, help="Default = 1. Determines the mean of random noise to add to the output labels of the model during training.")
    parser.add_argument('--noise_std', type=float, nargs='?', default=1.0, help="Default = 1. Determines the mean of random noise to add to the output labels of the model during training.")
    parser.add_argument('--patience_treshold', type=int, nargs='?', default=32, help="Default = 10. Sets the number of epochs to allow for training not to improve before early stopping.")
    #metrics to use for evaluation
    parser.add_argument('--metric', type=str, nargs='?', default="clustered_rmse", help="Default=accuracy. Defines the metric to use to select the best model")
    parser.add_argument('--additional_metrics', type=str, nargs='+', default= ['clustered_mse', 'clustered_rmse', 'clustered_mae', 'mse', 'rmse', 'mae'], help="Defaults to all ['clustered_mse', 'clustered_rmse', 'clustered_mae', 'mse', 'rmse', 'mae']")
    #interpretation options
    parser.add_argument('--compute_attributions', type=str2bool, nargs='?', const=True, default=False, help="Default=False. If compute_attributions, then attributions are computed and saved for each sample. If set to false, the interpretation routine loads saved attributions instead.")
    parser.add_argument('--interpretation_algorithm', type=str, default='DeepLift', help='Default "DeepLift". Defines interpretation algorithm to use in the interpretation routine (Saliency, IntegratedGradients, NoiseTunnel, DeepLift.')
    parser.add_argument('--interpretation_predict', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Generates predictions on input and logs them when performing interpretation routine")
    parser.add_argument('--generate_video_visualizations', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Generates interpretation videos for the generated attributions.")
    parser.add_argument('--generate_outstanding_frames_grid', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Generates interpretation grid of the frames with highest attributions.")
    parser.add_argument('--generate_action_unit_cross_correlation', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Action Unit (AU) cross-correlation score with attributions.")
    parser.add_argument('--generate_region_importances', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Generates attributions for certain face regions.")
    parser.add_argument('--visualization_methods', nargs='+', default=['heat_map'], help='Default [\"heat_map\"]. Visualization methods for the interpretation videos from original_image, heat_map, blended_heat_map, masked_image, alpha_scaling')
    parser.add_argument('--attribution_signs', nargs='+', default=['absolute_value'], help='Default [\"absolute_value\"]. The sign to use to visualize interpretation videos\' attribution from: absolute_value, all, positive, negative')
    #### Higher level model architecture to be fine-tuned 
    parser.add_argument('--model_name', type=str, nargs='?', default="r3d_18", help="Default=r3d_18. The model name to use (r3d_18, mc3_18, r2plus1d_18, CNN, RNN, CNNAttention).")
    parser.add_argument('--model_depth', type=int, nargs='?', help="No default. Sets the number of layers to use on 2018 video ResNet models")
    parser.add_argument('--pretrained_dataset', type=str, nargs='?', default='K', help="Default='k' (Kinnetics-700). Sets the dataset(s) that was/were used to pretrain the model weights. Options: ['K', 'M', 'S', 'KM', 'KMS']")
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help="Default=True. If true, loads pretrained weights to the model")
    #parser.add_argument('--pretrained_dataset', type=str, nargs='?', default='casia-webface', help="Default=casia-webface. The dataset used to pretrain facenet on")
    parser.add_argument('--num_classes', type=int, nargs='?', default=1, help="The number of output classes of the last layer of the model")
    #type of architecture to process sequential data
    args = parser.parse_args()
    #args = vars(parser.parse_args())
    print('args: ', args)
    #start experiment log
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Timestamp: ", timestamp)
    logger = Logger(timestamp, LOGGER_PATH)
    logger.log(**vars(args))
    #Initialize TensorBoard logger
    comment = f'_model_name={args.model_name}_model_depth={args.model_depth}_pretrained_dataset={args.pretrained_dataset}_window_size={args.window_size}_input_dilation={args.input_dilation}_overlap={args.overlap}_momentum={args.momentum}_weight_decay={args.weight_decay}_batch_size={batch_size}_epochs={args.epochs}_noise_mean={args.noise_mean}_noise_std={args.noise_std}_lr={args.lr}'
    writer = SummaryWriter(comment=comment)
    #Training/testing mode parameters
    if not args.train and not args.test and not args.interpret:
        warn("No training nor testing routines specified. Nothing to be done. Run python main.py [-h --help] to see the usage.")
    #Runtime parameters
    epochs = args.epochs
    #learning rate
    #max_lr = 1e-4
    lr = args.lr#2.5e-3#1e-4###1e-4 is the beset starting lr for default step scheduler
    min_lr = args.lr//100#1e-6
    momentum = args.momentum #0.9 #In practice 0, 0.5, 0.9 or 0.99
    clip_value = 1.0
    weight_decay = args.weight_decay #0.5#1e-3#1e-4 
    validation_metric_score = 0
    best_validation_metric_score = math.inf
    #run main procedure
    main()
    print('DONE')
