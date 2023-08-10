from share import *
import config

import os
import einops
import numpy as np
import torch
import random
import tensorrt as trt


from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from combine2 import merge


class hackathon():

    def to_trt(self):
        unet_model = self.model.model.diffusion_model

        x = torch.randn((1, 4, 32, 48), dtype=torch.float32, device='cuda')
        timesteps = torch.zeros(1, dtype=torch.int32, device='cuda')
        context = torch.randn((1, 77, 768), dtype=torch.float32, device='cuda')

        control1 = torch.randn((1, 320, 32, 48), dtype=torch.float32, device='cuda')
        control2 = torch.randn((1, 320, 32, 48), dtype=torch.float32, device='cuda')
        control3 = torch.randn((1, 320, 32, 48), dtype=torch.float32, device='cuda')
        control4 = torch.randn((1, 320, 16, 24), dtype=torch.float32, device='cuda')
        control5 = torch.randn((1, 640, 16, 24), dtype=torch.float32, device='cuda')
        control6 = torch.randn((1, 640, 16, 24), dtype=torch.float32, device='cuda')
        control7 = torch.randn((1, 640, 8, 12), dtype=torch.float32, device='cuda')
        control8 = torch.randn((1, 1280, 8, 12), dtype=torch.float32, device='cuda')
        control9 = torch.randn((1, 1280, 8, 12), dtype=torch.float32, device='cuda')
        control10 = torch.randn((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
        control11 = torch.randn((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
        control12 = torch.randn((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
        control13 = torch.randn((1, 1280, 4, 6), dtype=torch.float32, device='cuda')

        control = [control1, control2, control3, control4, control5, control6, control7, control8, control9, control10,
                   control11, control12, control13]

        intput_name = ["x", "timesteps", "context", "control1", "control2", "control3", "control4", "control5",
                       "control6",
                       "control7", "control8", "control9", "control10", "control11", "control12", "control13"]

        with torch.inference_mode():
            torch.onnx.export(unet_model, (x, timesteps, context, control), './unet/unet.onnx', opset_version=17,
                              input_names=intput_name, output_names=["output"])

        # os.system("onnxsim unet.onnx unetsim.onnx")
        # os.system("trtexec --onnx=./unet/unet.onnx --saveEngine=unet.trt  --fp16 --builderOptimizationLevel=5 --inputIOFormats=fp32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw")
        # os.system("trtexec --onnx=./unet/unet.onnx --saveEngine=unet.trt  --fp16  --inputIOFormats=fp32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw")

        control_model = self.model.control_model

        input1 = torch.randn((1, 4, 32, 48), dtype=torch.float32, device='cuda')
        input2 = torch.randn((1, 3, 256, 384), dtype=torch.float32, device='cuda')
        input3 = torch.zeros(1, dtype=torch.int32, device='cuda')
        input4 = torch.randn((1, 77, 768), dtype=torch.float32, device='cuda')

        output_names = []

        for i in range(13):
            output_names.append("out_" + str(i))

        # 导出
        with torch.inference_mode():
            torch.onnx.export(control_model, (input1, input2, input3, input4), 'controlnet.onnx', opset_version=17,
                              keep_initializers_as_inputs=True,
                              input_names=['x_in', "hint_in", "timesteps_in", "context_in"], output_names=output_names)

        os.system("onnxsim controlnet.onnx controlnetsim.onnx")
        # os.system("trtexec --onnx=controlnetsim.onnx --saveEngine=controlnet.trt  --fp16 --inputIOFormats=fp32:chw,fp32:chw,int32:chw,fp32:chw")

        # 合并
        merge("./controlnetsim.onnx", "./unet/unet.onnx")
        os.system("trtexec --onnx=./combine/combinesim.onnx --saveEngine=combine.trt  --builderOptimizationLevel=5 --fp16 --inputIOFormats=fp32:chw,fp32:chw,int32:chw,fp32:chw,fp32:chw,int32:chw,fp32:chw")

        transformer = self.model.cond_stage_model

        input = torch.zeros((1, 77), dtype=torch.int32, device="cuda")

        # 导出 input_names和output_names 带括号
        with torch.inference_mode():
            torch.onnx.export(transformer, input, 'clip.onnx', input_names=['input_ids'], output_names=['output'],
                              opset_version=17)

        # clip_fp16()
        os.system("onnxsim clip.onnx clipsim.onnx")
        os.system(
            "trtexec --onnx=clipsim.onnx --saveEngine=clip.trt --inputIOFormats=int32:chw")


        #
        # vae_model = self.model.first_stage_model
        #
        # vae_model.forward = vae_model.decode
        #
        # z = torch.randn((1, 4, 32, 48), dtype=torch.float32, device='cuda')
        #
        # with torch.inference_mode():
        #     torch.onnx.export(vae_model, z, 'vae.onnx', opset_version=17, input_names=['z'], output_names=['output'])
        #
        # os.system("onnxsim vae.onnx vaesim.onnx")
        # os.system("trtexec --onnx=vaesim.onnx --saveEngine=vae.trt --builderOptimizationLevel=5")

    def clip(self):
        with open("./clip.trt", 'rb') as f:
            engine_str = f.read()
        clip_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        clip_context = clip_engine.create_execution_context()

        clip_context.set_binding_shape(0, (1, 77))
        self.model.clip_context = clip_context  # 替换模型
    
    def vae(self):
        with open("./vae.trt", 'rb') as f:
            engine_str = f.read()
        vae_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        vae_context = vae_engine.create_execution_context()

        vae_context.set_binding_shape(0, (1, 4, 32, 48))
        self.model.vae_context = vae_context

    def controlnet(self):
        with open("./controlnet.trt", 'rb') as f:
            engine_str = f.read()
        controlnet_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        controlnet_context1 = controlnet_engine.create_execution_context()
        self.model.controlnet_context1 = controlnet_context1  # 替换模型
        controlnet_context2 = controlnet_engine.create_execution_context()
        self.model.controlnet_context2 = controlnet_context2  # 替换模型
        # nIO = controlnet_engine.num_io_tensors
        # lTensorName = [controlnet_engine.get_tensor_name(i) for i in range(nIO)]


    def unet(self):
        with open("./unet.trt", 'rb') as f:
            engine_str = f.read()
        unet_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        unet_context1 = unet_engine.create_execution_context()
        self.model.unet_context1 = unet_context1  # 替换模型
        unet_context2 = unet_engine.create_execution_context()
        self.model.unet_context2 = unet_context2  # 替换模型
        # nIO = unet_engine.num_io_tensors
        # lTensorName = [unet_engine.get_tensor_name(i) for i in range(nIO)]

    def combine(self):
        with open("./combine.trt", 'rb') as f:
            engine_str = f.read()
        combine_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        combine_context1 = combine_engine.create_execution_context()
        self.model.combine_context1 = combine_context1  # 替换模型
        combine_context2 = combine_engine.create_execution_context()
        self.model.combine_context2 = combine_context2  # 替换模型

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        # self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # self.stream1 = cudart.cudaStreamCreate()[1]
        # self.stream2 = cudart.cudaStreamCreate()[1]
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        self.to_trt()
        self.clip()
        self.combine()
        # self.vae()

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode,
                strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)


            cond = {"c_concat": [control],
                    "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}#加载过一次cond_stage_model clip
            un_cond = {"c_concat": None if guess_mode else [control],
                       "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}  # 加载过一次cond_stage_model
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            # 循环执行
            samples, intermediates = self.ddim_sampler.sample(10, num_samples,
                                                              shape, cond, verbose=False, eta=eta,
                                                              unconditional_guidance_scale=scale,
                                                              unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)  # 执行 first_stage_config // aev
            # print(x_samples, x_samples.shape)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                               255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]
        return results

