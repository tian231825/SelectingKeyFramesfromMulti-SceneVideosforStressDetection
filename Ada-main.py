# -*- encoding: utf-8 -*-
"""
@File    : Ada-main.py
@Time    : 2023/8/15 13:28
@Author  : junruitian
@Software: PyCharm
"""
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from torch import distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Modules
from Config import opt
from Configuration.util import evaluate_policy
from Modules.KFS_Module import KFS_Environment
from data_loader import Ada_Encoder

'''
执行指令：python -m torch.distributed.launch --nproc_per_node=4 emotion_attention_main.py --distribute True 
                    --max_epoch 100 --save_model True --batch_size 12
    nohup指令： # nohup python -m torch.distributed.launch --nproc_per_node=4 emotion_attention_main.py --batch_size 8
                        --distribute True --max_epoch 80 --save_model False > nohup_80.out 2>&1 &

'''


# 每个gpu上的梯度求平均
def average_gradients(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def fetch_action(env, batch):
    for idx in range(0, batch):
        a = env.action_space.sample()
        new_dimension_array = np.expand_dims(a, axis=0)  # (1, 300)
        if idx == 0:
            a_res = new_dimension_array
        else:
            # print(self.state.shape)
            a_res = np.concatenate((a_res, new_dimension_array), axis=0)
    return a_res


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.distribute:
        torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式

    # step 1: Create Environment
    EnvName = 'Stress_video_KFS'
    BriefEnvName = 'Stress_video_KFS'
    env = KFS_Environment(total_frame_size=opt.total_frame_size, initial_state=opt.initial_state,
                          group_video=opt.group_include_video)
    eval_env = KFS_Environment(total_frame_size=opt.total_frame_size, initial_state=opt.initial_state,
                               group_video=opt.group_include_video)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_e_steps = opt.max_episode_steps

    # step 2: Seed everything
    # torch.manual_seed(opt.seed)
    # env.seed(opt.seed)
    # env.action_space.seed(opt.seed)
    # eval_env.seed(opt.seed)
    # eval_env.action_space.seed(opt.seed)
    # np.random.seed(opt.seed)
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm: SACD', '  Env:', EnvName, '  state_dim:', state_dim, '  action_dim:',
          action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', max_e_steps, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter

        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/SACD_{}'.format(BriefEnvName) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # step3: Build model and Replay buffer
    if not os.path.exists('model'):
        os.mkdir('model')

    model_agent = getattr(Modules, opt.model_name)
    model = model_agent(action_dim=action_dim, state_dim=state_dim, gamma=opt.gamma, lr=opt.lr,
                        hide_shape=opt.hid_shape,
                        alpha=opt.alpha, batch_size=opt.batch_size, adaptive_alpha=opt.adaptive_alpha, dvc=opt.dvc)

    # step4: Data Construction
    train_data = Ada_Encoder(opt.data_root, train=True)
    test_data = Ada_Encoder(opt.data_root, test=True)
    all_data = Ada_Encoder(opt.data_root, all=True)
    if opt.distribute is True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=False,
                                                       num_workers=opt.num_workers, pin_memory=True,
                                                       drop_last=False,
                                                       sampler=train_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                                      num_workers=opt.num_workers, pin_memory=True, drop_last=False,
                                                      sampler=test_sampler)
    else:
        train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        all_dataloader = DataLoader(all_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    if opt.render:
        # 跳出step 不会被调用evaluate_policy
        score = evaluate_policy(eval_env, model, True, 5)
        print('EnvName:', BriefEnvName, 'seed:', opt.seed, 'score:', score)
    else:
        round = 0
        for i, (rep, label, index) in enumerate(all_dataloader):
            round = round + 1
            video_a = rep[0]
            video_b = rep[1]
            video_c = rep[2]
            input_a = Variable(video_a)
            input_b = Variable(video_b)
            input_c = Variable(video_c)
            # label: torch.Size([batch])
            label = Variable(label)
            index = Variable(index)
            if opt.use_gpu:
                input_a = input_a.cuda()
                input_b = input_b.cuda()
                input_c = input_c.cuda()
                label = label.cuda()
                index = index.cuda()
            if opt.distribute:
                # 获取 local_rank
                local_rank = torch.distributed.get_rank()
                input_a = input_a.cuda(local_rank, non_blocking=True)
                input_b = input_b.cuda(local_rank, non_blocking=True)
                input_c = input_c.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)
                index = index.cuda(local_rank, non_blocking=True)
            # input a/b/c torch.Size([24, 100, 768])
            batch = input_a.shape[0]
            # print(batch)
            # torch size [100,768] for each video frame array
            total_steps = 0
            while total_steps < opt.max_episode_steps:
                # Do not use opt.seed directly, or it can overfit to opt.seed
                s, info = env.reset(seed=env_seed, batch=batch)
                env_seed += 1
                done = False
                while not done:
                    # dw: dead&win; tr: truncated
                    # e-greedy exploration
                    if total_steps < opt.random_steps:
                        a = fetch_action(env=env, batch=batch)
                    else:
                        a = model.select_action(s, deterministic=False)
                    # print(s.shape, a.shape)
                    s_next, r, done, info = env.step(s, a, input_a, input_b, input_c)
                    print("step:", total_steps)

                    if done:
                        import copy
                        temp_ = copy.deepcopy(s_next)
                        temp_ = temp_.cpu().numpy()
                        np.savetxt("record_"+ str(round)+".txt", temp_, fmt='%.4f', delimiter=' ')

                    for b in range(0, batch):
                        model.replay_buffer[b].add(s[b], a[b], r[b], s_next[b], done)
                    s = s_next
                    '''update if its time'''
                    # train 50 times every 50 steps rather than 1 training per step. Better!
                    if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                        for j in range(opt.update_every):
                            model.train()
                    '''record & log'''
                    # if total_steps % opt.eval_interval == 0:
                    #     score = evaluate_policy(eval_env, model, video_a=input_a[idx], video_b=input_b[idx],
                    #                             video_c=input_c[idx])
                    #     if opt.write:
                    #         writer.add_scalar('ep_r', score, global_step=total_steps)
                    #         writer.add_scalar('alpha', model.alpha, global_step=total_steps)
                    #         writer.add_scalar('H_mean', model.H_mean, global_step=total_steps)
                    #     print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
                    #         'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                    total_steps += 1
                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        model.save(int(total_steps / 1000), BriefEnvName)
    env.close()
    eval_env.close()
