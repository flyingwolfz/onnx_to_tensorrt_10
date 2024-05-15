import tensorrt as trt
import os
import tools2.trt_tools as trt_tools
import torch
import tools2.tools as tools
import cv2
import numpy as np

logger= trt.Logger()

def get_engine(onnx_file_path, engine_file_path="",input_shape=(1, 3, 1472 * 2, 3072 * 2),force_rebuild=False):
    def build_engine():
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, logger)
        runtime = trt.Runtime(logger)

        # config.optimization_level = 5 #larger, can use more time to optimize,
        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) # 256MiB
        config.set_flag(trt.BuilderFlag.FP16)

        if not os.path.exists(onnx_file_path):
            print("ONNX file not found.")
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        network.get_input(0).shape = [*input_shape]
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)  # engine serialized
        engine = runtime.deserialize_cuda_engine(plan)  # engine deserialize
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

    if force_rebuild==True:
        return build_engine()
    else:
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()


def main():

    onnx_file_path = "onnx.onnx"
    engine_file_path = "trt.trt"
    force_rebuild = False
    save_path = 'trt_out.png'
    H=1472
    W=3072
    C=3
    output_shape = [(1, C, H, W)]
    input_shape = (1, C, H , W )

    #creat engine
    engine=get_engine(onnx_file_path, engine_file_path,input_shape=input_shape,force_rebuild=force_rebuild)

    #inference
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = trt_tools.allocate_buffers(engine)

    # zeros = np.zeros(shape=input_shape)
    # zeros = np.float32(zeros)
    # inputs[0].host = zeros

    load_path  = 'scene_116_gt.png'
    net_input =tools.load_img(path=load_path, color_channel=C, to_amp=1)
    net_input=net_input.cpu().data.numpy()
    net_input = np.float32(net_input)
    inputs[0].host = net_input

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_num = 10
    for i in range(run_num):
        trt_outputs = trt_tools.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs,
                                             stream=stream)
    end.record()
    torch.cuda.synchronize()
    print('time: (ms):', start.elapsed_time(end) / run_num)
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shape)]

    trt_outputs=trt_outputs[0]
    trt_outputs = torch.from_numpy(trt_outputs).cuda()
    trt_outputs= tools.phase_to_img_color(trt_outputs)
    cv2.imwrite(save_path, trt_outputs)

    # Free host and device memory used for inputs and outputs
    trt_tools.free_buffers(inputs, outputs, stream)

if __name__ == "__main__":
    main()

