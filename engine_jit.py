import math
import sys
import os

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def _resolve_fid_statistics_file(img_size):
    if img_size == 256:
        stats_name = "font_100_test_chars_fid_stats.npz"
    else:
        raise NotImplementedError(f"Unsupported img_size for FID stats: {img_size}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    fid_statistics_file = os.path.join(base_dir, "fid_stats", stats_name)
    if not os.path.exists(fid_statistics_file):
        raise FileNotFoundError(f"FID statistics file not found: {fid_statistics_file}")
    return fid_statistics_file


def _save_reference_images(target_images_all, num_images, ref_folder):
    """把 test_npz 里的 ground truth 图像保存到 ref_folder，供 torch-fidelity 0.3.0 用作 input2。

    torch-fidelity PyPI 0.3.0 不支持 fid_statistics_file 参数（那是 LTH14 fork 的特性），
    因此改为直接提供真实的参考图像目录。
    """
    os.makedirs(ref_folder, exist_ok=True)
    for img_id in range(num_images):
        out_path = os.path.join(ref_folder, '{:05d}.png'.format(img_id))
        if os.path.exists(out_path):
            continue  # 已生成则跳过，避免重复 I/O
        target_img = target_images_all[img_id].transpose([1, 2, 0])
        target_img = target_img[:, :, ::-1]  # RGB -> BGR for cv2
        cv2.imwrite(out_path, target_img)


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0

        font_labels, char_labels, style_images, content_images = labels

        style_images = style_images.to(device, non_blocking=True).to(torch.float32).div_(255)
        style_images = style_images * 2.0 - 1.0
        content_images = content_images.to(device, non_blocking=True).to(torch.float32).div_(255)
        content_images = content_images * 2.0 - 1.0

        labels = (
            font_labels.to(device, non_blocking=True),
            char_labels.to(device, non_blocking=True),
            style_images,
            content_images
        )

        with torch.amp.autocast('cuda', dtype=misc.get_amp_dtype()):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    test_data = np.load(args.test_npz_path)
    font_labels_all = test_data['font_labels']
    char_labels_all = test_data['char_labels']
    style_images_all = test_data['style_images']
    content_images_all = test_data['content_images']
    target_images_all = test_data['target_images']

    num_total_samples = len(font_labels_all)
    num_images = min(args.num_images, num_total_samples)
    num_steps = (num_images + batch_size * world_size - 1) // (batch_size * world_size)

    base_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], num_images, args.img_size
        )
    )
    if getattr(args, 'eval_step_folders', False):
        base_folder = os.path.join(base_folder, "step_{}".format(epoch))
    save_folder = os.path.join(base_folder, "compare")
    gen_folder = os.path.join(base_folder, "generated")
    print("Save to:", base_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(gen_folder, exist_ok=True)

    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        if start_idx >= num_images:
            torch.distributed.barrier()
            continue

        font_labels_gen = torch.from_numpy(font_labels_all[start_idx:end_idx]).long().cuda()
        char_labels_gen = torch.from_numpy(char_labels_all[start_idx:end_idx]).long().cuda()

        style_images_gen = torch.from_numpy(style_images_all[start_idx:end_idx].copy()).float().cuda()
        style_images_gen = style_images_gen / 255.0 * 2.0 - 1.0
        content_images_gen = torch.from_numpy(content_images_all[start_idx:end_idx].copy()).float().cuda()
        content_images_gen = content_images_gen / 255.0 * 2.0 - 1.0

        labels_gen = (font_labels_gen, char_labels_gen, style_images_gen, content_images_gen)

        with torch.amp.autocast('cuda', dtype=misc.get_amp_dtype()):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        pairs_buffer = []
        for b_id in range(sampled_images.size(0)):
            img_id = start_idx + b_id
            if img_id >= num_images:
                continue
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(gen_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

            target_img = target_images_all[img_id].transpose([1, 2, 0])
            target_img = target_img[:, :, ::-1]
            pair_img = np.concatenate([target_img, gen_img], axis=1)
            pairs_buffer.append((img_id, pair_img))

            if len(pairs_buffer) == 8:
                grid_id = pairs_buffer[0][0] // 8
                rows = []
                for row_idx in range(4):
                    left_pair = pairs_buffer[row_idx * 2][1]
                    right_pair = pairs_buffer[row_idx * 2 + 1][1]
                    row = np.concatenate([left_pair, right_pair], axis=1)
                    rows.append(row)
                grid_img = np.concatenate(rows, axis=0)
                cv2.imwrite(os.path.join(save_folder, 'grid_{}.png'.format(str(grid_id).zfill(5))), grid_img)
                pairs_buffer = []

        if len(pairs_buffer) > 0:
            grid_id = pairs_buffer[0][0] // 8
            img_h, img_w = pairs_buffer[0][1].shape[:2]
            while len(pairs_buffer) < 8:
                pairs_buffer.append((-1, np.zeros((img_h, img_w, 3), dtype=np.uint8)))
            rows = []
            for row_idx in range(4):
                left_pair = pairs_buffer[row_idx * 2][1]
                right_pair = pairs_buffer[row_idx * 2 + 1][1]
                row = np.concatenate([left_pair, right_pair], axis=1)
                rows.append(row)
            grid_img = np.concatenate(rows, axis=0)
            cv2.imwrite(os.path.join(save_folder, 'grid_{}.png'.format(str(grid_id).zfill(5))), grid_img)

    torch.distributed.barrier()

    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    if log_writer is not None:
        ref_folder = os.path.join(base_folder, "reference")
        _save_reference_images(target_images_all, num_images, ref_folder)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=gen_folder,
            input2=ref_folder,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    torch.distributed.barrier()


def train_one_epoch_single_gpu(model, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0

        font_labels, char_labels, style_images, content_images = labels

        style_images = style_images.to(device, non_blocking=True).to(torch.float32).div_(255)
        style_images = style_images * 2.0 - 1.0
        content_images = content_images.to(device, non_blocking=True).to(torch.float32).div_(255)
        content_images = content_images * 2.0 - 1.0

        labels = (
            font_labels.to(device, non_blocking=True),
            char_labels.to(device, non_blocking=True),
            style_images,
            content_images
        )

        with torch.amp.autocast('cuda', dtype=misc.get_amp_dtype()):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate_single_gpu(model, args, epoch, batch_size=64, log_writer=None):
    model.eval()

    test_data = np.load(args.test_npz_path)
    font_labels_all = test_data['font_labels']
    char_labels_all = test_data['char_labels']
    style_images_all = test_data['style_images']
    content_images_all = test_data['content_images']
    target_images_all = test_data['target_images']

    num_total_samples = len(font_labels_all)
    num_images = min(args.num_images, num_total_samples)
    num_steps = (num_images + batch_size - 1) // batch_size

    base_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model.method, model.steps, model.cfg_scale,
            model.cfg_interval[0], model.cfg_interval[1], num_images, args.img_size
        )
    )
    if getattr(args, 'eval_step_folders', False):
        base_folder = os.path.join(base_folder, "step_{}".format(epoch))
    save_folder = os.path.join(base_folder, "compare")
    gen_folder = os.path.join(base_folder, "generated")
    print("Save to:", base_folder)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(gen_folder, exist_ok=True)

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = batch_size * i
        end_idx = start_idx + batch_size

        font_labels_gen = torch.from_numpy(font_labels_all[start_idx:end_idx]).long().cuda()
        char_labels_gen = torch.from_numpy(char_labels_all[start_idx:end_idx]).long().cuda()

        style_images_gen = torch.from_numpy(style_images_all[start_idx:end_idx].copy()).float().cuda()
        style_images_gen = style_images_gen / 255.0 * 2.0 - 1.0
        content_images_gen = torch.from_numpy(content_images_all[start_idx:end_idx].copy()).float().cuda()
        content_images_gen = content_images_gen / 255.0 * 2.0 - 1.0

        labels_gen = (font_labels_gen, char_labels_gen, style_images_gen, content_images_gen)

        with torch.amp.autocast('cuda', dtype=misc.get_amp_dtype()):
            sampled_images = model.generate(labels_gen)

        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        pairs_buffer = []
        for b_id in range(sampled_images.size(0)):
            img_id = start_idx + b_id
            if img_id >= num_images:
                continue
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(gen_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

            target_img = target_images_all[img_id].transpose([1, 2, 0])
            target_img = target_img[:, :, ::-1]
            pair_img = np.concatenate([target_img, gen_img], axis=1)
            pairs_buffer.append((img_id, pair_img))

            if len(pairs_buffer) == 8:
                grid_id = pairs_buffer[0][0] // 8
                rows = []
                for row_idx in range(4):
                    left_pair = pairs_buffer[row_idx * 2][1]
                    right_pair = pairs_buffer[row_idx * 2 + 1][1]
                    row = np.concatenate([left_pair, right_pair], axis=1)
                    rows.append(row)
                grid_img = np.concatenate(rows, axis=0)
                cv2.imwrite(os.path.join(save_folder, 'grid_{}.png'.format(str(grid_id).zfill(5))), grid_img)
                pairs_buffer = []

        if len(pairs_buffer) > 0:
            grid_id = pairs_buffer[0][0] // 8
            img_h, img_w = pairs_buffer[0][1].shape[:2]
            while len(pairs_buffer) < 8:
                pairs_buffer.append((-1, np.zeros((img_h, img_w, 3), dtype=np.uint8)))
            rows = []
            for row_idx in range(4):
                left_pair = pairs_buffer[row_idx * 2][1]
                right_pair = pairs_buffer[row_idx * 2 + 1][1]
                row = np.concatenate([left_pair, right_pair], axis=1)
                rows.append(row)
            grid_img = np.concatenate(rows, axis=0)
            cv2.imwrite(os.path.join(save_folder, 'grid_{}.png'.format(str(grid_id).zfill(5))), grid_img)

    if log_writer is not None:
        ref_folder = os.path.join(base_folder, "reference")
        _save_reference_images(target_images_all, num_images, ref_folder)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=gen_folder,
            input2=ref_folder,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
