from model import *
from Trainer import *
import argparse

def main(args):
    print('Building the model')

    params = {
        'fusion': False,
        # 'train_loc': 1,
        # 'train_dim': 0,
        # 'train_theta': 0,
        # 'train_dir': 0
    }
    model = Model(graph=None, **params)

    print('Training')

    trainer = ModelTrainer(model=model, data_base_path=args.data_path)

    if args.num_samples is not None:
        args.num_samples = int(args.num_samples)
    if args.restore in ['True', 'true']:
        args.restore = True
    else:
        args.restore = False

    if args.augment in ['True', 'true']:
        args.augment = True
    else:
        args.augment = False

    if args.segmentation_kitti in ['True', 'true']:
        args.segmentation_kitti = True
    else:
        args.segmentation_kitti = False

    if args.segmentation_cityscapes in ['True', 'true']:
        args.segmentation_cityscapes = True
    else:
        args.segmentation_cityscapes = False

    params = {
        'batch_size': int(args.batch_size), 
        'epochs': int(args.epochs), 
        'random_seed': int(args.random_seed),
        'num_samples': args.num_samples,
        'save_steps': int(args.save_steps),
        'restore': args.restore,
        'training_per': float(args.training_per),
        'training': True,
        'epochs_img_head': int(args.epochs_img_head),
        'epochs_img_all': int(args.epochs_img_all),
        'segmentation_kitti': args.segmentation_kitti,
        'segmentation_cityscapes': args.segmentation_cityscapes,
        'num_summary_images': int(args.num_summary_images),
        'start_epoch': int(args.start_epoch),
        'augment': args.augment
    }
    if args.train_images_seg in ['True', 'true']:
        print('Train Segmentation')
        trainer.train_images_seg(**params)

    if args.train_bev in ['True', 'true']:
        print('Train Detection')
        trainer.train_bev(**params)

    if args.train_bev_lr_find in ['True', 'true']:
        print('Train Detection - LR Find')
        trainer.train_bev_lr_find(**params)


    if args.train_fusion in ['True', 'true']:
        print('Train Fusion')
        trainer.train_fusion(**params)

    if args.train_end_to_end in ['True', 'true']:
        print('Train end to end')
        trainer.train_end_to_end(**params)

   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Object Detection Model')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--epochs', default=100, help='epochs')
    parser.add_argument('--epochs_img_head', default=20, help='epochs')
    parser.add_argument('--epochs_img_all', default=50, help='epochs')
    parser.add_argument('--num_samples', default=None, help='num samples')
    parser.add_argument('--random_seed', default=0, help='random seed')
    parser.add_argument('--save_steps', default=100, help='save steps')
    parser.add_argument('--restore', default=False, help='restore')
    parser.add_argument('--training_per', default=0.5)
    parser.add_argument('--num_summary_images', default=4)
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--augment', default=True)
    parser.add_argument('--train_bev_lr_find', default=False)
    
    parser.add_argument('--train_end_to_end', default=False)

    parser.add_argument('--segmentation_kitti', default=False)
    parser.add_argument('--segmentation_cityscapes', default=True)

    parser.add_argument('--data_path', default='/Users/apple/Desktop/Master/Data', help='Data path')
    
    parser.add_argument('--train_bev', default=False, help='train bev')
    parser.add_argument('--train_images_seg', default=False, help='train image header - segmentation')
    parser.add_argument('--train_fusion', default=False, help='train fusion')
    
    args = parser.parse_args()

    main(args)

