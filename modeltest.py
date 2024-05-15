import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.net1(x)
        return out1


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.net1(x)
        return out1


class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False):
        super().__init__()
        if last_layer == True:
            self.net1 = nn.Sequential(
                nn.Conv2d(in_channels, 4 * out_channels, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                # nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            )
        else:
            self.net1 = nn.Sequential(
                nn.Conv2d(in_channels, 4 * out_channels, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                # nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.ReLU(),
            )

    def forward(self, x):
        out1 = self.net1(x)
        return out1





class Net2(nn.Module):
    def __init__(self, input_type, channel_num):
        super().__init__()
        self.channel_num = channel_num
        self.input_type = input_type
        first_channel = 32

        if self.input_type == '1view':
            self.first_layer = Conv(self.channel_num, first_channel)

        k = 64
        self.netdown2 = Down(first_channel, k)
        self.netdown3 = Down(k, k * 2)

        self.netup3 = Up(k * 2, k)
        self.netup4 = Up(k, self.channel_num * 2, last_layer=True)

    def forward(self, x):

        dout1 = self.first_layer(x)
        dout2 = self.netdown2(dout1)
        uout4 = self.netup4(dout2)
        if self.channel_num == 1:
            out = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0)  # single_color
        elif self.channel_num == 3:
            out0 = torch.atan2(uout4[:, 0, :, :], uout4[:, 1, :, :]).unsqueeze(0)
            out1 = torch.atan2(uout4[:, 2, :, :], uout4[:, 3, :, :]).unsqueeze(0)
            out2 = torch.atan2(uout4[:, 4, :, :], uout4[:, 5, :, :]).unsqueeze(0)
            out = torch.cat((out0, out1, out2), 1)  # bgr full color
        return out


if __name__ == "__main__":
    from thop import profile
    import tools.tools as tools

    import_onnx = 1
    import_name = 'onnx.onnx'
    load_trained = 0
    load_pth_name = '.pth'
    show_onnx = 0
    check_onnx = 1
    run_num = 10


    input_type = '1view'


    H = 1472
    W = 3072
    channel_num = 3  # 1 or 3

    net = Net2(input_type=input_type, channel_num=channel_num).cuda()
    net_input = tools.get_rand_input(input_type=input_type, W=W, H=H, channel_num=channel_num)

    if load_trained == 1:
        net.load_state_dict(torch.load(load_pth_name))

    flops, params = profile(net, inputs=net_input.unsqueeze(0))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops * 2 / 1e9))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(run_num):
            out = net(net_input)
    end.record()
    torch.cuda.synchronize()
    print('time: (ms):', start.elapsed_time(end) / run_num)

    if import_onnx == 1:
        import torch.onnx

        torch.onnx.export(net,  # model being run
                          net_input,  # model input
                          import_name,  # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=17,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          # verbose=True,
                          )
        print('import onnx,name:', import_name)
        # torch.onnx.export(net, net_input, 'trained_model/torchonnx.onnx', input_names=['input'],
        #                   output_names=['output'])  #same
    if check_onnx == 1:
        import onnx

        onnx.checker.check_model(onnx.load(import_name))
    if show_onnx == 1:
        import netron

        netron.start(import_name)