import importlib
import os
import time
from datetime import datetime
import random
import math
import torch
import numpy as np
from utils.setlogger import get_logger
import torch.nn as nn
from utils.losses import info_max_loss, Mask_CrossEntropyLabelSmooth, RDLoss
from utils.model_profiling import model_profiling  # 通过调用这个，初始化了一系列参数
from utils.config import FLAGS  # 此处加载参数
from utils.datasets import get_dataset
from itertools import cycle
from torch.autograd import Variable
from torch.autograd import Function
import warnings


warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

best_prec1 = 0.0

v_loss = []
acc_lst = []

# set log files， fixed 添加了时间戳
time_strip = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# * 保存路径
saved_path = os.path.join(FLAGS.log_dir, FLAGS.dataset, 'full', f'{FLAGS.exp}-{time_strip}')
if not os.path.exists(saved_path):  # 如果不存在就创建
    os.makedirs(saved_path)
logger = get_logger(
    os.path.join(
        saved_path,
        # * 标志是train 还是 test，以及从哪个域到哪个域
        '{}_{}2{}.log'.format('test' if FLAGS.test_only else 'train_lt', FLAGS.sdomain, FLAGS.tdomain),
    )
)  # 配置 log

all_nets = [[1, 1, 224], [1, 0.9, 224], [1, 1, 192], [1, 0.9, 192], [0.5, 1, 224], [1, 1, 160], [0.5, 0.9, 224], [1, 0.9, 160], [0.5, 1, 192], [1, 1, 128], [0.5, 0.9, 192], [1, 0.9, 128], [0.5, 1, 160], [0.5, 0.9, 160], [0.5, 1, 128], [0.5, 0.9, 128]]

# * 14种配置，除了 minnet 和 supernet
inter_subnets = all_nets[1:-1]


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
            FLAGS.lr,
            momentum=FLAGS.momentum,
            nesterov=FLAGS.nesterov,
            weight_decay=FLAGS.weight_decay,
        )
    return optimizer


def train(
    epoch,
    source_loader,
    target_loader,
    rd_loss,
    model_student,
    model_teacher,
    criterion,
    optimizer,
    lr_scheduler,
):
    # 得到now 的当下数值
    model_student.train()
    # 不停的配对使用
    combi_loader = zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader)
    tot_source_ce_loss = 0.0
    tot_rd_loss = 0.0
    tot_target_pl_loss = 0.0
    for batch_idx, data in enumerate(combi_loader):
        # try:
        (source_data, target_data) = data
        source_input_list, source_target = source_data
        source_target = source_target.cuda(non_blocking=True)
        # * 不是imagenet，没有训练 imagenet 的代码
        teacher_output_lst = []

        if FLAGS.dataset != 'imagenet1k':
            target_input_list, _ = target_data
            # ** 特质化 teacher 的最大模型
            # 将 teacher 模型的参数取最大。
            model_teacher.apply(lambda d: setattr(d, 'depth_mult', max(FLAGS.depth_mult_range)))  # * apply应该是对模型的每个属性和函数，传入当作参数
            model_teacher.apply(lambda m: setattr(m, 'width_mult', max(FLAGS.width_mult_range)))
            model_teacher.apply(lambda r: setattr(r, 'res', max(FLAGS.resolution_list)))
            with torch.no_grad():  # * 因为 teacher 是不需要梯度的
                # * 此处的 teacher 是最大网的输出，因为是没有乘小因子的
                teacher_output_max = model_teacher(target_input_list[0].cuda(non_blocking=True), dom=1)
                # * 对所有的 teacher的内部子网，得到输出，放入到 list 中
                for snt in inter_subnets:
                    # ** 因为不能将所有的网络都预先配制好，不然内存不够，所以运行的时候再配置 teacher
                    model_teacher.apply(lambda r: setattr(r, 'res', snt[2]))
                    model_teacher.apply(lambda m: setattr(m, 'width_mult', snt[1]))
                    model_teacher.apply(lambda d: setattr(d, 'depth_mult', snt[0]))
                    # ** 得到网络后，输入数据，得到中间层的预测。teacher 是需要每一个网络都进行配置的；而 student 只需要4个
                    # * 找到对应的 resolution 的 index，然后将数据输入
                    teacher_output_in = model_teacher(
                        target_input_list[FLAGS.resolution_list.index(snt[2])].cuda(non_blocking=True),
                        dom=1,
                    ).detach()
                    # * 将内网结果append到 list。teacher 没有最小网
                    teacher_output_lst.append(teacher_output_in)

        optimizer.zero_grad()
        # ** max-subnet, 最大的子网的构成，[supnet, 1, 2, minnet]
        # * 必须做成[[],]
        student_subnets = [
            [max(FLAGS.depth_mult_range), max(FLAGS.width_mult_range), max(FLAGS.resolution_list)],
        ]
        if epoch > FLAGS.warm_ep:
            # ** 2 random subnet，中间子网，随机选2个
            for _ in range((FLAGS.num_subnets - 2)):
                # 可以随意改中间子网的数量，随机选择两个子网作为中间网
                student_subnets.append(random.choice(inter_subnets))
        if epoch > FLAGS.warm_ep2:
            # ** 最小的 student 网。min-subnet
            student_subnets.append(
                [
                    min(FLAGS.depth_mult_range),
                    min(FLAGS.width_mult_range),
                    min(FLAGS.resolution_list),
                ]
            )
        subnet_out_lst = []
        info_loss = torch.tensor(0.0).cuda(non_blocking=True)
        for sn in student_subnets:
            t_start = time.time()
            # 对于 student 的子网来说，#* 一般来说是 4 个
            model_student.apply(lambda d: setattr(d, 'depth_mult', sn[0]))
            model_student.apply(lambda m: setattr(m, 'width_mult', sn[1]))
            model_student.apply(lambda r: setattr(r, 'res', sn[2]))
            # 如果是最大网，对齐参数
            if sn[0] == max(FLAGS.depth_mult_range) and sn[1] == max(FLAGS.width_mult_range) and sn[2] == max(FLAGS.resolution_list):
                # print('maxnet')
                # ** NB！！，取出相应的数据； 因为输入的数据是 4 种分辨率，所以必须根据当前的分辨率来选择这个数据！
                # 选择输入
                maxnet_output = model_student(
                    source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=0,
                )
                source_ce_loss = FLAGS.gamma_ce * criterion(maxnet_output, source_target).cuda(non_blocking=True)  # we use λcls = 1, 15, 64, λrd = 1, 1, 0.5 for Office-31, Office-Home and DomainNet, respectively,λpl = 0.1 for all the datasets.
                tot_source_ce_loss += source_ce_loss / FLAGS.gamma_ce

                target_rd_loss = torch.tensor(0.0).cuda(non_blocking=True)

                # 得到 maxnet 的输出，将maxnet的输出作为伪标签
                maxnet_output_target = model_student(
                    target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=1,
                )
                maxnet_output_target_detach = maxnet_output_target.detach()
                # * 通过最大网的输出 softmax，得到伪标签
                pseudo_label = torch.softmax(maxnet_output_target_detach, dim=-1)
                # 输出（values, indices)，取最大
                max_probs, targets_pl = torch.max(pseudo_label, dim=-1)
                # ** 如果伪标签中最大概率大于阈值，就会取
                mask = max_probs.ge(FLAGS.pl_thresh).float()
                # 包裹，也没有设置 requires_grad=True,默认是false 的
                targets_pl = torch.autograd.Variable(targets_pl)
                # target pl loss #** pl 是伪标签的缩写
                # 如果使用 .计算 maxnet 和 target 的误差  sum([criterion(maxnet_output_target[i], targets_pl[i]) for i in range(len(targets_pl)) if mask[i]])
                target_pl_loss = FLAGS.gamma_pl * criterion(maxnet_output_target, targets_pl, mask=mask).cuda(non_blocking=True)
                info_loss = torch.tensor(0.0).cuda(non_blocking=True)
                if FLAGS.use_iml:
                    info_loss = info_max_loss(torch.softmax(maxnet_output_target, dim=-1))
                

                im_pl_loss = target_pl_loss + info_loss.cuda(non_blocking=True)
                # ** 最大网是不需要这个的 gamma_pl 的
                tot_target_pl_loss += target_pl_loss / FLAGS.gamma_pl

                # 使用ADV
                # if FLAGS.use_discriminator:
                #     total_loss = source_ce_loss + source_adv_loss + target_adv_loss + im_pl_loss
                # else:
                #     total_loss = source_ce_loss + im_pl_loss

                total_loss = source_ce_loss + im_pl_loss
                total_loss.backward()

            # * 如果是minnet
            elif sn[0] == min(FLAGS.depth_mult_range) and sn[1] == min(FLAGS.width_mult_range) and sn[2] == min(FLAGS.resolution_list):
                minnet_output = model_student(
                    source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=0,
                )
                source_ce_loss = FLAGS.gamma_ce * criterion(minnet_output, source_target).cuda(non_blocking=True)
                tot_source_ce_loss += source_ce_loss

                minnet_output_target = model_student(
                    target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=1,
                )

                if FLAGS.use_iml:
                    info_loss = info_max_loss(torch.softmax(minnet_output_target, dim=-1))
                    target_rd_loss = FLAGS.gamma_rd * rd_loss(
                        minnet_output_target,
                        # * 取到平均值，作为结果给student 的minnet 学习
                        torch.mean(torch.stack(teacher_output_lst), dim=0),
                        epoch,
                    ).cuda(non_blocking=True) + info_loss.cuda(non_blocking=True)
                else:
                    target_rd_loss = FLAGS.gamma_rd * rd_loss(
                        minnet_output_target,
                        torch.mean(torch.stack(teacher_output_lst), dim=0),
                        epoch,
                    ).cuda(non_blocking=True)

                tot_rd_loss += (target_rd_loss - info_loss.cuda(non_blocking=True)) / FLAGS.gamma_rd

                total_loss = source_ce_loss + target_rd_loss

                total_loss.backward()

            # * student 中间的2个随机网络
            else:
                subnet_out = model_student(
                    source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=0,
                )
                source_ce_loss = FLAGS.gamma_ce * criterion(subnet_out, source_target).cuda(non_blocking=True)
                tot_source_ce_loss += source_ce_loss
                subnet_out_target = model_student(
                    target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=1,
                )

                if FLAGS.use_iml:
                    info_loss = info_max_loss(torch.softmax(subnet_out_target, dim=-1))
                    # ** 此处较 maxnet 多了 recursive distillation
                    target_rd_loss = FLAGS.gamma_rd * rd_loss(subnet_out_target, teacher_output_max, epoch).cuda(non_blocking=True) + info_loss.cuda(non_blocking=True)
                else:
                    target_rd_loss = FLAGS.gamma_rd * rd_loss(subnet_out_target, teacher_output_max, epoch).cuda(non_blocking=True)
                # 去掉 info_loss
                tot_rd_loss += (target_rd_loss - info_loss.cuda(non_blocking=True)) / FLAGS.gamma_rd

                total_loss = source_ce_loss + target_rd_loss
                total_loss.backward()

            # 每个模型都更新么teacher？对应参数更新更新 teacher 网，使用 EMA
            m = FLAGS.ema_decay  # m = 0.9
            for param_q, param_k in zip(model_student.parameters(), model_teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.data)  # 更新之
            cost_time = time.time() - t_start
            logger.info(
                f"Epoch:{epoch}/{FLAGS.num_epochs} Iter:{batch_idx}/{FLAGS.exp}[s:{len(source_loader)} t:{len(target_loader)}] Time: {cost_time:.3f}s LR:{optimizer.param_groups[0]['lr']:.6f} Subnet(DxWxR):{sn[0]:.1f}x{sn[1]:.1f}x{sn[2]} :: source_ce: {(source_ce_loss.item() / FLAGS.gamma_ce):.4f} target_rd: {((target_rd_loss.item() - info_loss.item()) / FLAGS.gamma_rd):.4f} target_pl: {(target_pl_loss.item() / FLAGS.gamma_pl):.4f} info_loss: {info_loss.item():.4f}"
            )
        # * student的子网都计算完毕后，开始更新参数
        optimizer.step()
        if not FLAGS.exp.count("LR_False"):
            if epoch > 15:
                lr_scheduler.step()


def validate(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size
    with torch.no_grad():
        for width_mult in [1.0]:
            model.apply(lambda r: setattr(r, 'res', resolution))
            model.apply(lambda d: setattr(d, 'depth_mult', 1))
            model.apply(lambda m: setattr(m, 'width_mult', 1.0))
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                output = model(
                    input[FLAGS.resolution_list.index(resolution)].cuda(non_blocking=True),
                    # ? 对于 warmup，是不要用source 来测
                    dom=1,
                )
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info(f'VAL {time.time() - t_start:.1f}s Epoch:{epoch}/{FLAGS.num_epochs} Loss:{loss/cnt:.4f} Acc:{acc/cnt:.3f}')
    v_loss.append(loss / cnt)
    acc_lst.append(acc / cnt)
    return acc / cnt


def test(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        for sn in all_nets:
            model.apply(lambda r: setattr(r, 'res', sn[2]))
            model.apply(lambda m: setattr(m, 'width_mult', sn[1]))
            model.apply(lambda d: setattr(d, 'depth_mult', sn[0]))

            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                output = model(
                    input[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),
                    dom=1,
                )
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info(f'TEST {(time.time() - t_start):6.1f}s Subnet(DxWxR):{sn[0]:.1f}x{sn[1]:.1f}x{sn[2]}, Epoch:{epoch}/{FLAGS.num_epochs} Loss:{loss/cnt:.4f} Acc:{(acc / cnt) * 100:.1f}')


def train_val_test():
    '''train and val'''
    global best_prec1
    # seed
    set_random_seed()  # 设置 seed
    # model，得到同样的 model
    model_student = get_model()
    model_teacher = get_model()
    model_student_wrapper = torch.nn.DataParallel(model_student).cuda()  # 数据并行
    model_teacher_wrapper = torch.nn.DataParallel(model_teacher).cuda()

    if FLAGS.lbl_smooth:  # 使用标签平滑，引入因子，缩小差距
        criterion = Mask_CrossEntropyLabelSmooth().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    # ** target_loader 和 val_loader 都是 target域的数据
    source_loader, target_loader, val_loader = get_dataset()

    # * 对于s_resnet     check ?，是否是对训练好的进行的呢？否则怎么会有 model_student 呢？
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        assert type(checkpoint) == dict and 'model_student' in checkpoint, print('Model is not OK')

        if type(checkpoint) == dict and 'model_student' in checkpoint:
            loaded_checkpoint_student = checkpoint['model_student']  # * 将 warmup 好的参数拷贝给 student 和 teacher
            loaded_checkpoint_teacher = checkpoint['model_student']  # not teacher

        # ** 对当前模型的 key 进行遍历，如果此 key 没在 loaded_ckp 中，就在 load_ckp中添加此 key，并用初始化值进行初始化
        for k_new in model_student_wrapper.state_dict().keys():
            if k_new not in loaded_checkpoint_student:
                loaded_checkpoint_student[k_new] = model_student_wrapper.state_dict()[k_new]
        model_student_wrapper.load_state_dict(loaded_checkpoint_student, strict=True)

        for k_new in model_teacher_wrapper.state_dict().keys():
            if k_new not in loaded_checkpoint_teacher:
                loaded_checkpoint_teacher[k_new] = model_teacher_wrapper.state_dict()[k_new]
        model_teacher_wrapper.load_state_dict(loaded_checkpoint_teacher, strict=True)
        print('Loaded model from {}.'.format(FLAGS.pretrained))

    optimizer = get_optimizer(model_student_wrapper)
    # * 取较大者 check resume training
    loader_size = len(source_loader) if len(source_loader) > len(target_loader) else len(target_loader)  # 取最大即可。可以直接写 max(len(source_loader), len(target_loader))
    if FLAGS.resume:  # 没有使用 resume
        checkpoint = torch.load(FLAGS.resume)
        model_student_wrapper.load_state_dict(checkpoint['model_student'])
        model_teacher_wrapper.load_state_dict(checkpoint['model_student'])
        last_epoch = checkpoint['last_epoch']
        optimizer.param_groups[0]['lr']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size * FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(FLAGS.resume, last_epoch))
    else:
        # ** 通常设为总步数的 10%
        # warm_up_iter = int(FLAGS.num_epochs * 0.1)
        # lr_max = FLAGS.lr
        # lr_min = FLAGS.lr * 0.05  # * 最终的是20分之一的原始的
        # T_max = FLAGS.num_epochs
        # # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        # lambda0 = lambda cur_iter: ((cur_iter + 5) / (warm_up_iter + 5) if cur_iter < warm_up_iter else (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / FLAGS.lr)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size * FLAGS.num_epochs)  # * 对每个 epoch 的每一个 loader 进行更新
        last_epoch = lr_scheduler.last_epoch  # 取出最后一个 epoch

    # ============ preparing loss ... ============
    rd_loss = RDLoss(
        FLAGS.num_classes,
        FLAGS.warmup_teacher_temp,
        FLAGS.teacher_temp,
        FLAGS.warmup_teacher_temp_epochs,
        FLAGS.num_epochs,
    ).cuda()

    if FLAGS.test_only:
        logger.info('Start testing.'.center(100, '$'))
        print(f'{FLAGS.num_epochs}eps_soep{FLAGS.sonly_ep}_lr{FLAGS.lr}_bs{FLAGS.batch_size}_gc{FLAGS.gamma_ce}gd{FLAGS.gamma_rd}_{FLAGS.sdomain}to{FLAGS.tdomain}.pt')
        test(last_epoch, val_loader, model_student_wrapper, criterion, source_loader)  # * test
        return

    logger.info('Start training.'.center(100, '*'))
    for epoch in range(last_epoch + 1, FLAGS.num_epochs + 1):
        # train
        train(
            epoch,
            source_loader,
            target_loader,
            rd_loss,
            model_student_wrapper,
            model_teacher_wrapper,
            criterion,
            optimizer,
            lr_scheduler,
        )
        # val
        print('Student Acc:'.center(100, '^'))
        prec1 = validate(epoch, val_loader, model_student_wrapper, criterion, source_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        output_best = '%03d Best Prec@1: %.3f' % (epoch, best_prec1 * 100)
        print(output_best.center(100, '&'))
        if epoch > FLAGS.warm_ep:
            test(epoch, val_loader, model_student_wrapper, criterion, source_loader)
        if is_best:
            torch.save(
                {
                    'model_student': model_student_wrapper.state_dict(),
                    'model_teacher': model_teacher_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                },
                # * 保存路径
                os.path.join(saved_path, f'ckpt_bestin{FLAGS.num_epochs}eps_soep{FLAGS.sonly_ep}_lr{FLAGS.lr}_bs{FLAGS.s_bs}x{FLAGS.t_bs}_gc{FLAGS.gamma_ce}gd{FLAGS.gamma_rd}gpl{FLAGS.gamma_pl}_sd_{FLAGS.random_seed}_{FLAGS.exp}.pt'),
            )

    logger.info(f'{FLAGS.num_epochs}eps_soep{FLAGS.sonly_ep}_lr{FLAGS.lr}_bs{FLAGS.batch_size}_gc{FLAGS.gamma_ce}gd{FLAGS.gamma_rd}gpl{FLAGS.gamma_pl}_{FLAGS.exp}.pt')

    logger.info('{} --> {}'.format(FLAGS.sdomain, FLAGS.tdomain))
    # ** test student & teacher
    # test(epoch, val_loader, model_student_wrapper, criterion, source_loader)  # student 网络
    logger.info('Teacher:')
    test(epoch, val_loader, model_teacher_wrapper, criterion, source_loader)  # teacher 网络

    return


def main():
    logger.info(FLAGS)
    '''train and eval model'''
    train_val_test()


if __name__ == '__main__':
    main()
