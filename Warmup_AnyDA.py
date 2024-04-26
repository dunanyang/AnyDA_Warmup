import importlib
import os
import time
from datetime import datetime
import random
import torch
import numpy as np
from utils.setlogger import get_logger
import torch.nn as nn
from utils.config import FLAGS  # 此处加载参数
from utils.datasets import get_dataset
import warnings

warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

global_step = 0
best_prec1 = 0
s_running_loss = 0.0
t_running_loss = 0.0
s_loss_vals = []
tpl_loss_vals = []
td_loss_vals = []
v_loss = []
acc_lst = []
# set log files， fixed 添加了时间戳
time_strip = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


all_nets = [[1, 1, 224], [1, 0.9, 224], [1, 1, 192], [1, 0.9, 192], [0.5, 1, 224], [1, 1, 160], [0.5, 0.9, 224], [1, 0.9, 160], [0.5, 1, 192], [1, 1, 128], [0.5, 0.9, 192], [1, 0.9, 128], [0.5, 1, 160], [0.5, 0.9, 160], [0.5, 1, 128], [0.5, 0.9, 128]]

# * 14种配置，除了 minnet 和 supernet
inter_subnets = all_nets[1:-1]



# * 保存路径
saved_path = os.path.join(FLAGS.log_dir, FLAGS.dataset, 'warmup', f'{FLAGS.exp}-{time_strip}')
if not os.path.exists(saved_path):  # 如果不存在就创建
    os.makedirs(saved_path)
logger = get_logger(
    os.path.join(
        saved_path,
        # * 标志是train 还是 test，以及从哪个域到哪个域
        '{}_{}2{}.log'.format('test' if FLAGS.test_only else 'train_lt', FLAGS.sdomain, FLAGS.tdomain),
    )
)  # 配置 log


def set_random_seed():
    '''set random seed'''
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model():
    '''get model'''
    model_lib = importlib.import_module(FLAGS.model)  # 使用 s_resnet
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)  # 实例化类，得到对象。输入 cls 和 size，实例化
    return model


def get_optimizer(model):
    '''get optimizer'''
    # all depthwise convolution (N, 1, x, x) has no weight decay
    # weight decay only on normal conv and fc
    # * 如果是，就要加可学习参数了
    if FLAGS.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {
                'params': params,
                'weight_decay': weight_decay,
                'lr': FLAGS.lr,
                'momentum': FLAGS.momentum,
                'nesterov': FLAGS.nesterov,
            }
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    # * 对于不是 ImageNet1k 的，就直接优化
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            FLAGS.lr,  # * 0.0002
            momentum=FLAGS.momentum,
            nesterov=FLAGS.nesterov,
            weight_decay=FLAGS.weight_decay,  # 0.001
        )
    return optimizer


def train(epoch, source_loader, target_loader, model_student, criterion, optimizer, lr_scheduler):
    model_student.train()
    len_s, len_t = len(source_loader), len(target_loader)

    # ** 只对source_target 进行训练
    for batch_idx, (inputs, targets) in enumerate(source_loader):
        source_target = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        for_train_nets = [[sorted(FLAGS.depth_mult_range)[-1], sorted(FLAGS.width_mult_range)[-1], sorted(FLAGS.resolution_list)[-1]]]

        # ** 当一定 epoch 之后，加入2 random subnet
        if epoch >= FLAGS.warm_ep:
            for _ in range((FLAGS.num_subnets - 2)):
                for_train_nets.append(random.choice(inter_subnets))
            for_train_nets.append([sorted(FLAGS.depth_mult_range)[0], sorted(FLAGS.width_mult_range)[0], sorted(FLAGS.resolution_list)[0]])
        for sn in for_train_nets:
            t_start = time.time()
            model_student.apply(lambda d: setattr(d, 'depth_mult', sn[0]))
            model_student.apply(lambda m: setattr(m, 'width_mult', sn[1]))
            model_student.apply(lambda r: setattr(r, 'res', sn[2]))

            # ** 根据 resolution 大小，得到相应的 index，然后取出输入；都输入最大的 resolution
            input_x = inputs[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True)

            output = model_student(input_x, dom=0)
            source_ce_loss = criterion(output, source_target).cuda(non_blocking=True)
            total_loss = source_ce_loss
            total_loss.backward()

            cost_time = time.time() - t_start
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch:{epoch}/{FLAGS.num_epochs} Iter:{batch_idx}/{FLAGS.exp}[s:{len_s} t:{len_t}] Time: {cost_time:.3f}s LR:{lr:.6f} Subnet(DxWxR):{sn[0]:.1f}x{sn[1]:.1f}x{sn[2]} source_ce:{source_ce_loss.item():.4f}'
            )
        # ** 优化器迭代
        optimizer.step()

        # if epoch > FLAGS.warm_ep:
        #     lr_scheduler.step()



class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=31, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = FLAGS.num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # Cross Entropy loss after smoothing the labels
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        # * 进行了 soft
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)

        return loss


def validate(epoch, loader, model, criterion):
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size
    with torch.no_grad():
        for width_mult in [1.0]:
            model.apply(lambda r: setattr(r, 'res', resolution))
            model.apply(lambda d: setattr(d, 'depth_mult', 1))
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                output = model(input[FLAGS.resolution_list.index(resolution)].cuda(non_blocking=True), dom=0)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('VAL {:.1f}s {}x Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(time.time() - t_start, str(width_mult), epoch, FLAGS.num_epochs, loss / cnt, acc / cnt))
    v_loss.append(loss / cnt)
    acc_lst.append(acc / cnt)
    return acc / cnt


# * 这里的 loader 是 val_loader； 而postloader 是 source_loader
def test(epoch, loader, model: nn.Module, criterion, postloader):
    t_start = time.time()
    model.eval()
    from models.slimmable_ops import SwitchableBatchNorm2d

    with torch.no_grad():

        # ** 对每个子网进行 test
        for sn in all_nets:
            model.apply(lambda r: setattr(r, 'res', sn[2]))
            model.apply(lambda m: setattr(m, 'width_mult', sn[1]))
            model.apply(lambda d: setattr(d, 'depth_mult', sn[0]))

            for n, m in model.named_modules():
                # * 恢复BN 的状态？
                if isinstance(m, SwitchableBatchNorm2d):
                    for bn in m.bn:
                        bn: torch.nn.BatchNorm2d
                        bn.reset_running_stats()
                        # * 这里是为了统计数据，所以设为 training mode
                        bn.training = True
                        bn.momentum = None
            # ？ 在 source_loader上输出60步？可能是为了 BN 的参数？可能是的
            for batch_id, (input_targ, _) in enumerate(postloader):
                model(input_targ[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True), dom=0)
                if batch_id > 60:
                    break

            # ** 开始eval 模式
            model.eval()

            loss, acc, cnt = 0, 0, 0
            # ** 这里的 loader 是 val_loader 的数据
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)

                # bug 这里有问题
                output = model(input[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True), dom=0)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('TEST {:.1f}s Subnet(DxWxR):{:.1f}x{:.1f}x{} Epoch:{}/{} Loss:{:.4f} Acc:{:.1f}'.format(time.time() - t_start, sn[0], sn[1], sn[2], epoch, FLAGS.num_epochs, loss / cnt, (acc / cnt) * 100))


def train_val_test():
    '''train and val'''
    global best_prec1
    # seed
    set_random_seed()  # 设置 seed
    # model，得到同样的 model
    model_student = get_model()
    # mark # ** 加载 imagenet 权重
    model_student.load_imagenet_weights()
    model_student.cuda()

    model_student_wrapper = torch.nn.DataParallel(model_student)  # 数据并行

    if FLAGS.lbl_smooth:  # 使用标签平滑，引入因子，缩小差距
        criterion = CrossEntropyLabelSmooth().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    # ** target_loader 和 val_loader 都是 target域的数据
    source_loader, target_loader, val_loader = get_dataset()

    optimizer = get_optimizer(model_student_wrapper)
    # check resume training
    loader_size = max(len(source_loader), len(target_loader))  # 取最大即可。可以直接写 max(len(source_loader), len(target_loader))
    if FLAGS.resume:  # 没有使用 resume
        checkpoint = torch.load(FLAGS.resume)
        model_student_wrapper.load_state_dict(checkpoint['model_student'])
        last_epoch = checkpoint['last_epoch']
        optimizer.param_groups[0]['lr']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size * FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(FLAGS.resume, last_epoch))
    else:
        # warm_up_iter = FLAGS.warm_ep * 0.1
        # lr_max = FLAGS.lr
        # lr_min = FLAGS.lr * 0.05  # * 最终的是20分之一的原始的
        # T_max = FLAGS.num_epochs
        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        # lambda0 = lambda cur_iter: ((cur_iter + 5) / (warm_up_iter + 5) if cur_iter < warm_up_iter else (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / FLAGS.lr)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size * FLAGS.num_epochs)  # * 对每个 epoch 的每一个 loader 进行更新
        last_epoch = lr_scheduler.last_epoch  # 取出最后一个 epoch

    logger.info('\n' + 'Start Warmup.'.center(100, '*'))
    for epoch in range(last_epoch + 1, FLAGS.num_epochs + 1):
        # train
        train(epoch, source_loader, target_loader, model_student_wrapper, criterion, optimizer, lr_scheduler)
        # val
        print(f'{epoch} Student Acc:'.center(100, '-'))
        # * 使用 target 数据集进行测试
        prec1 = validate(epoch, val_loader, model_student_wrapper, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        output_best = 'Best Prec@1: %.3f' % (best_prec1 * 100)
        print(output_best.center(100, '#'))

        if epoch > 30 and epoch % 3 == 0:  # * 每15个 epoch，test 一下
            test(epoch, val_loader, model_student_wrapper, criterion, source_loader)

        # if is_best:
        torch.save(
            {
                'model_student': model_student_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
            },
            os.path.join(
                saved_path,
                f'final_{FLAGS.sdomain}2{FLAGS.tdomain}.pt',
            ),
        )
        if is_best:
            torch.save(
                {
                    'model_student': model_student_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                },
                os.path.join(
                    saved_path,
                    f'best_{FLAGS.sdomain}2{FLAGS.tdomain}.pt',
                ),
            )

    print('{}eps_soep{}_lr{}_bs{}_gc{}gd{}gpl{}_{}2{}.pt'.format(FLAGS.num_epochs, FLAGS.sonly_ep, FLAGS.lr, FLAGS.batch_size, FLAGS.gamma_ce, FLAGS.gamma_rd, FLAGS.gamma_pl, FLAGS.sdomain, FLAGS.tdomain))
    print('{} --> {}'.format(FLAGS.sdomain, FLAGS.tdomain))
    # ** 最后测试一下
    test(last_epoch, val_loader, model_student_wrapper, criterion, source_loader)
    return


def main():
    logger.info(FLAGS)
    '''train and eval model'''
    train_val_test()


if __name__ == '__main__':

    main()
