import torch
import numpy as np
import visdom
import time
import torch.nn.modules as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from Model.encoder_model import MetricNet
from Model.advDA_net import DomainClassifier, AdversarialNetwork
from my_utils.training_utils import sample_task_tr, sample_task_te
from my_utils.init_utils import weights_init0, weights_init1, weights_init2, set_seed
from my_utils.visualize_utils import Visualize_v2
from Data_generate.DataLoadFn import DataGenFn
from lib import *
import pandas as pd

device = torch.device('cuda:0')
vis = visdom.Visdom(env='yancy_env')
generator = DataGenFn()

# ====== hyper params =======
CHN = 1
DIM = 1024  # 2048
CHECK_EPOCH = 10
WEIGHT_DECAY = 0  # 1e-5
CHECK_D = False  # check domain adaptation by t-SNE
Load = [3,2,1,0]
# WEIGHTS_INIT = weights_init2  # best for DASMN
# WEIGHTS_INIT = weights_init1
WEIGHTS_INIT = weights_init0  # better

adda_params = dict(alpha=1, gamma=10)
running_params = dict(train_epochs=150, test_epochs=3,
                      train_episodes=30, test_episodes=100,
                      train_split=100, test_split=0,  # generally same with train_split.
                      )
# ==========================


class DASMNLearner:
    def __init__(self, n_way, n_support, n_query):
        self.way = n_way
        self.ns = n_support
        self.nq = n_query
        self.visualization = Visualize_v2(vis)
        self.domain_criterion = nn.NLLLoss() #标准

        self.model = MetricNet(self.way, self.ns, self.nq, vis=vis, cb_num=8).to(device)
        self.d_classifier = DomainClassifier(DIM=DIM).to(device)
        self.d_classifier_separate = DomainClassifier(DIM=DIM).to(device)
        self.adv_cls = AdversarialNetwork(256).to(device)

    @staticmethod
    def get_constant(episodes, ep, epi):
        # total_steps = epochs * episodes
        total_steps = 3000  # 1000
        start_steps = ep * episodes
        p = float(epi + start_steps) / total_steps
        constant = torch.tensor(2. / (1. + np.exp(-adda_params['gamma'] * p)) - 1).to(device)
        return constant

    def domain_loss(self, src_x, tar_x, constant, draw=False):
        s_feature = self.model.get_features(src_x.reshape([-1, CHN, DIM]))
        t_feature = self.model.get_features(tar_x.reshape([-1, CHN, DIM]))
        src_dy, s_d_feat = self.d_classifier(s_feature, constant)
        tar_dy, t_d_feat = self.d_classifier(t_feature, constant)

        row_sum = torch.sum(tar_dy, dim=1)
        normalized_log = torch.div(tar_dy, row_sum.unsqueeze(1))

        index0 = torch.max(normalized_log, 1)[0] > 0.3
        logit1_demo = tar_dy[index0, :]
        fc1_t_demo = t_d_feat[index0, :]

        new_index = torch.max(normalized_log, 1)[0] - torch.min(normalized_log, 1)[0] < 0.03
        new_logit1_demo=tar_dy[new_index,:]
        new_fc1_t_demo=t_d_feat[new_index,:]
        new_temp=2+new_logit1_demo
        l_open=nn.ReLU(inplace=False)(new_temp)

        l_open=l_open.sum(dim=1)
        l_open=l_open.mean()

        data2_index_0 = torch.max(normalized_log, 1)[0] <= 0.3
        data2_index_1 = torch.max(normalized_log, 1)[0] - torch.min(normalized_log, 1)[0] >= 0.03
        data2_index = data2_index_0 & data2_index_1
        data2_logit1 = tar_dy[data2_index, :]
        data2_fc1_t = t_d_feat[data2_index, :]

        module_new = nn.Softmax(dim=-1)
        predict_prob_train = module_new(src_dy,)
        predict_prob_test1_demo = module_new(logit1_demo)

        prob_discriminator_train = self.adv_cls.forward(s_d_feat)

        prob_discriminator_test1_demo = self.adv_cls.forward(fc1_t_demo)

        data2_prob_discriminator__ = self.adv_cls.forward(data2_fc1_t)

        prob_discriminator_train_separate = self.adv_cls.forward(s_d_feat.detach())

        prob_discriminator_test1_demo_separate = self.adv_cls.forward(fc1_t_demo.detach())

        new_prob_discriminator_test3_demo_separate = self.adv_cls.forward(new_fc1_t_demo.detach())

        data2_prob_discriminator___separate = self.adv_cls.forward(data2_fc1_t.detach())

        omega_train = get_omega_s(prob_discriminator_train_separate, src_dy, domain_temperature=1.0,
                                  class_temperature=10.0)
        omega_train = normalize_weight(omega_train)

        omega_test1 = get_omega_s(prob_discriminator_test1_demo_separate, logit1_demo,
                                  domain_temperature=1.0, class_temperature=10.0)
        omega_test1 = normalize_weight(omega_test1)
        # == data2_omega *

        data2_omega = get_omega_t(data2_prob_discriminator___separate, data2_logit1,
                                  domain_temperature=1.0, class_temperature=1.0)
        data2_omega = normalize_weight(data2_omega)

        adv_loss = torch.zeros(1, 1).to(device)
        adv_loss_separate = torch.zeros(1, 1).to(device)

        tmp = omega_test1 * nn.BCEWithLogitsLoss(reduction='none')(prob_discriminator_test1_demo,
                                                         torch.ones_like(prob_discriminator_test1_demo))

        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        tmp = omega_train * nn.BCEWithLogitsLoss(reduction='none')(prob_discriminator_train,
                                                         torch.ones_like(prob_discriminator_train))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        temp = data2_omega * nn.BCEWithLogitsLoss(reduction='none')(data2_prob_discriminator__,
                                                          torch.zeros_like(data2_prob_discriminator__))
        adv_loss += torch.mean(temp, dim=0, keepdim=True)

        adv_loss_separate += nn.BCEWithLogitsLoss()(prob_discriminator_train_separate,
                                          torch.ones_like(prob_discriminator_train_separate))

        adv_loss_separate += nn.BCEWithLogitsLoss()(prob_discriminator_test1_demo_separate,
                                          torch.ones_like(prob_discriminator_test1_demo_separate))

        adv_loss_separate += nn.BCEWithLogitsLoss()(new_prob_discriminator_test3_demo_separate,
                                          torch.zeros_like(new_prob_discriminator_test3_demo_separate))

        open_loss = adv_loss + adv_loss_separate  + l_open

        s_domain_y = torch.zeros(src_dy.shape[0]).long().to(device)
        t_domain_y = torch.ones(tar_dy.shape[0]).long().to(device)
        s_domain_loss = self.domain_criterion(src_dy, s_domain_y)
        t_domain_loss = self.domain_criterion(tar_dy, t_domain_y)
        s_domain_acc = torch.eq(src_dy.max(-1)[1], s_domain_y).float().mean()
        t_domain_acc = torch.eq(tar_dy.max(-1)[1], t_domain_y).float().mean()

        if draw:
            pass

        return (s_domain_loss, t_domain_loss, open_loss), (s_domain_acc, t_domain_acc)

    def joint_training(self, src_tasks, tgt_tasks):
        self.model.train(), self.d_classifier.train()
        self.model.apply(WEIGHTS_INIT), self.d_classifier.apply(WEIGHTS_INIT)

        c_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        d_optimizer = torch.optim.RMSprop(self.d_classifier.parameters(), lr=1e-3, alpha=0.99)
        # d_optimizer = torch.optim.Adam(self.d_classifier.parameters(), lr=1e-3, weight_decay=1e-5)
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.95)  # lr=lr∗gamma^epoch
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.95)  # lr=lr∗gamma^epoch

        tar_tr = tgt_tasks[:, :src_tasks.shape[1]]  # N_src = N_tgt
        print('source set for training:', src_tasks.shape)
        print('target set for training', tar_tr.shape)
        print('target set for validation', tgt_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['train_epochs']
        episodes = running_params['train_episodes']
        counter = 0
        draw = False
        avg_ls = torch.zeros([episodes])
        times = np.zeros([epochs])

        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        for ep in range(epochs):
            # if (ep + 1) <= 3 and CHECK_D:
            #     draw = True
            # elif 25 <= (ep + 1) <= 40 and CHECK_D:
            #     draw = True

            delta = 10 if (ep + 1) <= 30 else 5
            t0 = time.time()
            for epi in range(episodes):
                support, query = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)
                tgt_s, _ = sample_task_tr(tar_tr, self.way, self.ns, length=DIM)
                tgt_v_s, tgt_v_q = sample_task_tr(tgt_tasks, self.way, self.ns, length=DIM)

                src_loss, src_acc, _ = self.model.forward(xs=support, xq=query, sne_state=False)
                constant = self.get_constant(episodes, ep, epi)
                domain_loss, domain_acc = self.domain_loss(support, tgt_s, constant, draw=draw)
                draw = False

                d_loss = domain_loss[0] + domain_loss[1] + adda_params['alpha'] * domain_loss[2]
                loss = src_loss + adda_params['alpha'] * d_loss

                c_optimizer.zero_grad()
                d_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                # nn.utils.clip_grad_norm_(parameters=self.d_classifier.parameters(), max_norm=0.5)
                # To clip the grads of d_classifier is Not recommended.
                c_optimizer.step()
                d_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    tgt_loss, tgt_acc, _ = self.model.forward(xs=tgt_v_s, xq=tgt_v_q, sne_state=False)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls


            # epoch
            t1 = time.time()
            times[ep] = t1 - t0
            print('[epoch {}/{}] time: {:.6f} Total: {:.6f}'.format(ep + 1, epochs, times[ep], np.sum(times)))
            ls_ = torch.mean(avg_ls).cpu()  # .item()
            print('[epoch {}/{}] avg_loss: {:.6f}\n'.format(ep + 1, epochs, ls_))
            if isinstance(c_optimizer, torch.optim.SGD):
                c_scheduler.step()  # ep // 5
            d_scheduler.step()  # ep // 5

            if ep + 1 >= CHECK_EPOCH and (ep + 1) % delta == 0:
                flag = input("Shall we stop the training? Y/N\n")
                flag = flag == 'y' or flag == 'Y'
                if flag:
                    print('Training stops!(manually)')
                    break

        print("The total time: {:.5f} s\n".format(np.sum(times)))

    def joint_training_2op(self, src_tasks, tgt_tasks, model_path):
        self.model.train(), self.d_classifier.train()
        self.model.apply(WEIGHTS_INIT), self.d_classifier.apply(WEIGHTS_INIT)

        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=1e-4)  # SGD is better
        c_optimizer = optimizer1  # for encoder
        d_optimizer = torch.optim.RMSprop(self.d_classifier.parameters(), lr=1e-3, alpha=0.999)
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
        # =======

        optional_lr = 0.01
        # =======

        tar_tr = tgt_tasks[:, :src_tasks.shape[1]]
        print('source set for training:', src_tasks.shape)
        print('target set for training', tar_tr.shape)
        print('target set for validation', tgt_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['train_epochs']
        episodes = running_params['train_episodes']
        counter = 0
        draw = False
        opt_flag = False
        avg_ls = torch.zeros([episodes])
        times = np.zeros([epochs])

        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        for ep in range(epochs):


            delta = 10 if (ep + 1) <= 30 else 5
            t0 = time.time()
            for epi in range(episodes):
                support, query = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)
                tgt_s, _ = sample_task_tr(tar_tr, self.way, self.ns, length=DIM)
                tgt_v_s, tgt_v_q = sample_task_tr(tgt_tasks, self.way, self.ns, length=DIM)

                src_loss, src_acc, _, _, _ = self.model.forward(xs=support, xq=query, sne_state=False)
                constant = self.get_constant(episodes, ep, epi)
                domain_loss, domain_acc = self.domain_loss(support, tgt_s, constant, draw=draw)
                draw = False

                d_loss = domain_loss[0] + domain_loss[1] + adda_params['alpha'] * domain_loss[2]
                loss = src_loss + adda_params['alpha'] * d_loss

                c_optimizer.zero_grad()
                d_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                # nn.utils.clip_grad_norm_(parameters=self.d_classifier.parameters(), max_norm=0.5)
                # To clip the grads of d_classifier is Not recommended.
                c_optimizer.step()
                d_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    tgt_loss, tgt_acc, _, _, _ = self.model.forward(xs=tgt_v_s, xq=tgt_v_q, sne_state=False)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls

            # epoch
            t1 = time.time()
            times[ep] = t1 - t0
            print('[epoch {}/{}] time: {:.5f} Total: {:.5f}'.format(ep + 1, epochs, times[ep], np.sum(times)))
            ls_ = torch.mean(avg_ls).cpu()  # .item()
            print('[epoch {}/{}] avg_loss: {:.8f}\n'.format(ep + 1, epochs, ls_))
            if isinstance(c_optimizer, torch.optim.SGD):
                c_scheduler.step()  # ep // 5
            d_scheduler.step()  # ep // 5
            if ls_ < optional_lr and opt_flag is False:
                # if (ep + 1) >= 20 and opt_flag is False:
                c_optimizer = optimizer1
                #     # c_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                print('====== Optimizer Switch ======\n')
                opt_flag = True

            if ep + 1 >= CHECK_EPOCH and (ep + 1) % delta == 0:
                flag = input("Shall we stop the training? Y/N\n")
                if flag == 'y' or flag == 'Y':
                    print('Training stops!(manually)')
                    new_path = os.path.join(model_path, f"final_epoch{ep+1}")
                    self.save(new_path, running_params['train_epochs'])
                    break
                else:
                    flag = input(f"Save model at epoch {ep+1}? Y/N\n")
                    if flag == 'y' or flag == 'Y':
                        child_path = os.path.join(model_path, f"epoch{ep+1}")
                        self.save(child_path, ep+1)

            # self.visualization.plot(data=[1000 * optimizer.param_groups[0]['lr']],
            #                         label=['LR(*0.001)'], counter=ep,
            #                         scenario="SSMN_Dynamic params")
        print("The total time: {:.5f} s\n".format(np.sum(times)))

    def test(self, tar_tasks, src_tasks=None, mask=False, model_eval=True):
        """
        :param mask:
        :param src_tasks: for t-sne
        :param tar_tasks: target tasks [way, n, dim]
        :return:
        """
        if model_eval:
            self.model.eval()
        else:
            self.model.train()
        print('target set', tar_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['test_epochs']
        episodes = running_params['test_episodes']
        # episodes = tar_tasks.shape[1] // self.ns
        print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(epochs, episodes,
                                                                           episodes * epochs))
        counter = 0
        avg_acc_all = 0.
        avg_loss_all = 0.

        print('Model.eval() is:', not self.model.training)

        for ep in range(epochs):
            avg_acc_ep = 0.
            avg_loss_ep = 0.
            sne_state = True
            for epi in range(episodes):
                tar_s, tar_q = sample_task_te(tar_tasks, self.way, self.ns, length=DIM)
                if src_tasks is not None and self.ns > 1:
                    src_s, src_q = sample_task_te(src_tasks, self.way, self.ns, length=DIM)

                # sne_state = True if epi + 1 == episodes else False
                with torch.no_grad():
                    tar_loss, tar_acc, zq_t,_,_ = self.model.forward(xs=tar_s, xq=tar_q, sne_state=sne_state)
                    if src_tasks is not None and self.ns > 1:
                        _, _, zq_s,_,_ = self.model.forward(xs=src_s, xq=src_s, sne_state=False)
                        if mask:
                            _, _, zq_t,_,_ = self.model.forward(xs=src_q, xq=src_q, sne_state=False)

                tar_ls, tar_ac = tar_loss.cpu().item(), tar_acc.cpu().item()
                avg_acc_ep += tar_ac
                avg_loss_ep += tar_ls

                self.visualization.plot([tar_ac, tar_ls], ['Acc', 'Loss'],
                                        counter=counter, scenario="CTFDA-Test")
                counter += 1
            # epoch
            avg_acc_ep /= episodes
            avg_loss_ep /= episodes
            avg_acc_all += avg_acc_ep
            avg_loss_all += avg_loss_ep
            print(f'[epoch {ep + 1}/{epochs}] avg_loss: {avg_loss_ep:.8f}\tavg_acc: {avg_acc_ep:.8f}')
        avg_acc_all /= epochs
        avg_loss_all /= epochs
        print('\n------------------------Average Result----------------------------')
        print('Average Test Loss: {:.6f}'.format(avg_loss_all))
        print('Average Test Accuracy: {:.6f}\n'.format(avg_acc_all))
        vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not self.model.training, avg_acc_all),
                 win='Eval:{} Test result'.format(not self.model.training))

    def save(self, filename, epoch):
        if os.path.exists(filename):
            filename += '(1)'
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'discriminator': self.d_classifier.state_dict(),
                 }
        torch.save(state, filename)
        print('This model is saved at [%s]' % filename)

    def load(self, filename, e=True, d=False):
        state = torch.load(filename)
        if e:
            self.model.load_state_dict(state['model_state'])
            print('Load Encoder successfully from [%s]' % filename)
        if d:
            self.d_classifier.load_state_dict(state['discriminator'])
            print('Load discriminator successfully from [%s]' % filename)


def train_operate(way, ns, nq, save_path, final_test=True, load_path=None):
    set_seed(120)
    nets = DASMNLearner(n_way=way, n_support=ns, n_query=nq)
    if load_path is not None:  # 若加载路径不为空，则默认：模型微调
        nets.load(load_path)

###########################################################################################

    # src, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                    label=False, data_len=DIM)
    # _, tar = generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
    #                                    label=False, data_len=DIM)

    # src, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, tar = generator.IMS_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    src, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
                                        label=False, data_len=DIM)
    _, tar = generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
                                       label=False, data_len=DIM)

    # src, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, tar = generator.IMS_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    # src, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, tar = generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    # src, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, tar = generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
    #                                    label=False, data_len=DIM)

    # training 1:
    print('Train joint_training')  # 推荐 train with 2 optimizers
    nets.joint_training_2op(src, tar, save_path)  # turn on the GRL


    if final_test:
        print('We test the model!')
        nets.test(src_tasks= None,tar_tasks=tar,  model_eval=True)
        nets.test(src_tasks= None,tar_tasks=tar,  model_eval=False)


def test_operate(way, ns, nq, path, eval_stats, ob_domain=False, num_domain=30):
    set_seed(283)

    model = DASMNLearner(n_way=way, n_support=ns, n_query=nq)

    # src_tasks, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                    label=False, data_len=DIM)
    # _, test_tasks = generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
    #                                     label=False, data_len=DIM)

    # src_tasks, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, test_tasks= generator.IMS_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    src_tasks, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
                                        label=False, data_len=DIM)
    _, test_tasks= generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
                                       label=False, data_len=DIM)

    # src_tasks, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, test_tasks= generator.IMS_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    # src_tasks, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, test_tasks= generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
    #                                    label=False, data_len=DIM)

    # src_tasks, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3, normalize=True,
    #                                     label=False, data_len=DIM)
    # _, test_tasks= generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
    #                                    label=False, data_len=DIM)


    # src_tasks = src_tasks if ob_domain else None
    running_params['test_episodes'] = 10 if ob_domain else 100
    # if you do not want to observe the src and tgt results together
    print('test_task shape: ', test_tasks.shape)
    if ob_domain:
        print('src_task shape: ', src_tasks.shape)

    if eval_stats == 'yes':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=True)
    elif eval_stats == 'both':
        model.load(path)
        print(src_tasks.shape)
        model.test(test_tasks, src_tasks, model_eval=True)
        print('\n================Reloading the file====================')
        model.load(path)
        # Attention! Model.train() would change the trained weights(it's an invalid operation),
        # so we have to reload the trained file again.
        model.test(test_tasks, src_tasks, model_eval=False)

    elif eval_stats == 'no':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=False)


if __name__ == "__main__":
    import os

    n_cls = 3  # 10, 3
    ns = nq = 5
    m_f_root = r"E:\CTFDA\Modelsave\CTFDA"

    # train:
    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        # save_path = os.path.join(m_f_root, 'dasmn_cw3to0_' + str(running_params['train_split']))
        save_path = os.path.join(m_f_root, 'CTFDA_funnySAnCLtest_otwtoims_' + str(running_params['train_split']))
        if not os.path.exists(m_f_root):  # 首先判断路径存在与否
            print(f'File path does not exist:\n{m_f_root}')
            exit()
        if os.path.exists(save_path):
            print(f'File path [{save_path}] exist!')
            order = input(f'Go on? y/n\n')
            if order == 'y' or order == 'Y':
                save_path += '(1)'
                print(f"Go on with file\n{save_path}")
            else:
                exit()

        os.mkdir(save_path)  # 如果要在训练过程中保存多个模型文件，请先建立文件夹
        train_operate(way=n_cls, ns=ns, nq=nq, save_path=save_path, final_test=True)

    # test:
    flag = input('Test? y/n\n')
    if flag == 'y' or flag == 'Y':
        load_path = r"E:\CTFDA\Modelsave\CTFDA\CTFDA_funnySAnCLtest_otwtoims_100\final_epoch35"
        test_operate(way=n_cls, ns=ns, nq=nq, path=load_path, eval_stats='both', ob_domain=True)
