import os
import tqdm
import wandb
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from data import PhysicsFleXDataset, FluidLabDataset
from data import prepare_input, get_scene_info, get_env_group, pad_relation_set
from models import Model, ChamferLoss
from utils import set_seed, AverageMeter, get_lr, Tee
from utils import count_parameters, my_collate


def main():
    args = gen_args()
    set_seed(args.random_seed)

    os.system("mkdir -p " + args.dataf)
    os.system("mkdir -p " + args.outf)

    tee = Tee(os.path.join(args.outf, "train.log"), "w")

    wandb.init(project="vgpl-training", config=args, name=args.env)

    ### training

    # load training data
    phases = ["train"] if args.eval == 0 else ["valid"]
    # phases = ["train"]
    if args.env in ["LatteArt", "Pouring"]:
        datasets = {phase: FluidLabDataset(args, phase) for phase in phases}
    else:
        datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}
    for phase in phases:
        if args.gen_data:
            datasets[phase].gen_data(args.env)
        else:
            datasets[phase].load_data(args.env)

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=args.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=args.num_workers,
            collate_fn=my_collate,
        )
        for phase in phases
    }

    # create model and train
    use_gpu = torch.cuda.is_available()
    model = Model(args, use_gpu)

    print("model #params: %d" % count_parameters(model))

    # checkpoint to reload model from
    model_path = None

    # resume training of a saved model (if given)
    if args.resume == 0:
        print("Randomly initialize the model's parameters")

    elif args.resume == 1:
        model_path = os.path.join(
            args.outf,
            "net_epoch_%d_iter_%d.pth" % (args.resume_epoch, args.resume_iter),
        )
        print("Loading saved ckp from %s" % model_path)

        if args.stage == "dy":
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # only load parameters in dynamics_predictor
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if "dynamics_predictor" in k and k in model_dict
            }
            model.load_state_dict(pretrained_dict, strict=False)

    # optimizer
    if args.stage == "dy":
        params = model.dynamics_predictor.parameters()
    else:
        raise AssertionError("unknown stage: %s" % args.stage)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    else:
        raise AssertionError("unknown optimizer: %s" % args.optimizer)

    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=3, verbose=True
    )

    # define loss
    particle_dist_loss = torch.nn.L1Loss()

    if use_gpu:
        model = model.cuda()

    # log args
    print(args)
    datasets["train"][3]
    # start training
    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf

    for epoch in range(st_epoch, args.n_epoch):
        for phase in phases:
            model.train(phase == "train")

            meter_loss = AverageMeter()
            meter_loss_raw = AverageMeter()

            meter_loss_ref = AverageMeter()
            meter_loss_nxt = AverageMeter()

            meter_loss_param = AverageMeter()

            # bar = ProgressBar(maxval=len(dataloaders[phase]))
            for i, data in tqdm.tqdm(enumerate(dataloaders[phase])):
                if args.stage == "dy":
                    # attrs: B x (n_p + n_s) x attr_dim
                    # particles: B x seq_length x (n_p + n_s) x state_dim
                    # n_particles: B
                    # n_shapes: B
                    # scene_params: B x param_dim
                    # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                    (
                        attrs,
                        particles,
                        n_particles,
                        n_shapes,
                        scene_params,
                        Rrs,
                        Rss,
                    ) = data

                    if use_gpu:
                        attrs = attrs.cuda()
                        particles = particles.cuda()
                        Rrs, Rss = Rrs.cuda(), Rss.cuda()

                    # statistics
                    B = attrs.size(0)
                    n_particle = n_particles[0].item()
                    n_shape = n_shapes[0].item()
                    state_dim = particles.shape[-1]

                    # p_rigid: B x n_instance
                    # p_instance: B x n_particle x n_instance
                    # physics_param: B x n_particle
                    groups_gt = get_env_group(
                        args, n_particle, scene_params, use_gpu=use_gpu
                    )

                    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                    # for now, only used as a placeholder
                    memory_init = model.init_memory(B, n_particle + n_shape)
                    num_future_preds = args.sequence_length - args.n_his
                    with torch.set_grad_enabled(phase == "train"):
                        preds = torch.zeros(
                            (num_future_preds, B, n_particle + n_shape, state_dim)
                        ).cuda()
                        gts = torch.zeros(
                            (num_future_preds, B, n_particle + n_shape, state_dim)
                        ).cuda()
                        preds_motion = torch.zeros(
                            (num_future_preds, B, n_particle + n_shape, state_dim)
                        ).cuda()
                        gts_motion = torch.zeros(
                            (num_future_preds, B, n_particle + n_shape, state_dim)
                        ).cuda()
                        for j in range(num_future_preds):
                            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                            state_cur = particles[:, : args.n_his]
                            # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                            Rr_cur = Rrs[:, args.n_his - 1]
                            Rs_cur = Rss[:, args.n_his - 1]

                            # predict the velocity at the next time step
                            inputs = [
                                attrs,
                                state_cur,
                                Rr_cur,
                                Rs_cur,
                                memory_init,
                                groups_gt,
                            ]

                            # pred_pos (unnormalized): B x n_p x state_dim
                            # pred_motion_norm (normalized): B x n_p x state_dim
                            pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

                            # concatenate the state of the shapes
                            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                            gt_pos = particles[:, args.n_his].clone()
                            pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

                            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            gt_motion = (
                                particles[:, args.n_his] - particles[:, args.n_his - 1]
                            )
                            mean_d, std_d = model.stat[2:]
                            gt_motion_norm = (gt_motion - mean_d) / std_d
                            pred_motion_norm = torch.cat(
                                [pred_motion_norm, gt_motion_norm[:, n_particle:]], 1
                            )
                            # insert the most recent prediction's info in the appropriate spot
                            particles[:, args.n_his] = pred_pos.clone()
                            R_batch = []
                            # Updating the batched relation sets (gotta account for padding as well)
                            for k in range(B):
                                Rrs_expanded = list(torch.unbind(Rrs[k]))
                                Rss_expanded = list(torch.unbind(Rss[k]))
                                (
                                    _,
                                    _,
                                    Rrs_expanded[args.n_his],
                                    Rss_expanded[args.n_his],
                                    _,
                                ) = prepare_input(
                                    pred_pos[k].detach().cpu().numpy(),
                                    n_particle,
                                    n_shape,
                                    args,
                                )
                                Rr_new, Rs_new = pad_relation_set(
                                    Rrs_expanded, Rss_expanded, n_particle, n_shape
                                )
                                R_batch.append([Rr_new, Rs_new])
                            Rrs, Rss = my_collate(R_batch)
                            Rrs = Rrs.cuda()
                            Rss = Rss.cuda()
                            # eject the oldest frame
                            # effectively we're swapping out the ground truth frame w/ our pred frame
                            particles = particles[:, 1:]
                            Rrs = Rrs[:, 1:]
                            Rss = Rss[:, 1:]
                            # insert our most recent pred in the set of multi-frame preds.
                            preds[j] = pred_pos
                            gts[j] = gt_pos
                            preds_motion[j] = pred_motion_norm
                            gts_motion[j] = gt_motion_norm
                        loss = F.l1_loss(
                            preds_motion[:, :, :n_particle],
                            gts_motion[:, :, :n_particle],
                        )
                        loss_raw = F.l1_loss(preds, gts)

                        meter_loss.update(loss.item(), B)
                        meter_loss_raw.update(loss_raw.item(), B)

                    if i % args.log_per_iter == 0:
                        print()
                        print(
                            "%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)"
                            % (
                                phase,
                                epoch,
                                args.n_epoch,
                                i,
                                len(dataloaders[phase]),
                                get_lr(optimizer),
                                loss.item(),
                                meter_loss.avg,
                                loss_raw.item(),
                                meter_loss_raw.avg,
                            )
                        )
                        wandb.log(
                            {
                                "loss": loss.item(),
                                "meter_loss": meter_loss.avg,
                                "loss_raw": loss_raw.item(),
                                "meter_loss_raw_avg": meter_loss_raw.avg,
                                "valid_loss": best_valid_loss,
                            }
                        )

                # update model parameters
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase == "train" and i > 0 and i % args.ckp_per_iter == 0:
                    model_path = "%s/net_epoch_%d_iter_%d.pth" % (args.outf, epoch, i)
                    torch.save(model.state_dict(), model_path)

            print(
                "%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f"
                % (phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
            )

            if phase == "valid" and not args.eval:
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    torch.save(model.state_dict(), "%s/net_best.pth" % (args.outf))


if __name__ == "__main__":
    main()
