import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv
from cbam import CBAM

class GCNModel(nn.Module):
    def __init__(self, args, mv):
        super(GCNModel, self).__init__()

        self.mv = mv
        self.args = args
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)

        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)

        #self.linear_x_1 = nn.Linear(self.fg, 256)
        #self.linear_x_2 = nn.Linear(256, 128)
        #self.linear_x_3 = nn.Linear(128, 64)

        #self.linear_y_1 = nn.Linear(self.fd, 256)
        #self.linear_y_2 = nn.Linear(256, 128)
        #self.linear_y_3 = nn.Linear(128, 64)
        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))

        if self.mv == 1:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 2:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 3:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 4:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 5:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 6:
            self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 7:
            self.cbamx = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 8:
            self.cbamx = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        elif self.mv == 9:
            self.cbamx = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cbamy = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)
            self.cnn_x = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fm, 1),
                                   stride=1,
                                   bias=True)
            self.cnn_y = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                                   out_channels=self.args.out_channels,
                                   kernel_size=(self.args.fd, 1),
                                   stride=1,
                                   bias=True)
        else:
            raise NameError("(mv) is not included in multiView index!")
        # self.cbamx = CBAM(self.args.view*self.args.gcn_layers, 5)
        # self.cbamy = CBAM(self.args.view*self.args.gcn_layers, 5)
        # self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        # self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))
        # self.fc1_x = nn.Linear(in_features=self.args.view*self.args.gcn_layers,
        #                      out_features=5*self.args.view*self.args.gcn_layers)
        # self.fc2_x = nn.Linear(in_features=5*self.args.view*self.args.gcn_layers,
        #                      out_features=self.args.view*self.args.gcn_layers)
        #
        # self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
        #                      out_features=5 * self.args.view * self.args.gcn_layers)
        # self.fc2_y = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
        #                      out_features=self.args.view * self.args.gcn_layers)
        #
        # self.sigmoidx = nn.Sigmoid()
        # self.sigmoidy = nn.Sigmoid()
        #

        # self.cnn_x = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
        #                        out_channels=self.args.out_channels,
        #                        kernel_size=(self.args.fm, 1),
        #                        stride=1,
        #                        bias=True)
        # self.cnn_y = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
        #                        out_channels=self.args.out_channels,
        #                        kernel_size=(self.args.fd, 1),
        #                        stride=1,
        #                        bias=True)

    def forward(self, data):

        torch.manual_seed(1)
        x_m = torch.randn(self.args.miRNA_number, self.args.fm)
        x_d = torch.randn(self.args.disease_number, self.args.fd)

        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_m_g1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_m_g2 = torch.relu(self.gcn_x2_s(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))

        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))

        y_d_g1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gcn_y2_s(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        if self.mv == 1:
            XM = torch.cat((x_m_f1, x_m_f2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 2:
            XM = torch.cat((x_m_f1, x_m_f2), 1).t()
            YD = torch.cat((y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 3:
            XM = torch.cat((x_m_f1, x_m_f2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 4:
            XM = torch.cat((x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 5:
            XM = torch.cat((x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 6:
            XM = torch.cat((x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 7:
            XM = torch.cat((x_m_f1, x_m_f2, x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2), 1).t()
            XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 8:
            XM = torch.cat((x_m_f1, x_m_f2, x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, 1 * self.args.gcn_layers, self.args.fd, -1)
        elif self.mv == 9:
            XM = torch.cat((x_m_f1, x_m_f2, x_m_g1, x_m_g2), 1).t()
            YD = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
            XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
            YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
        else:
            raise NameError("(mv) is not included in multiView index!")

        XM_channel_attention = self.cbamx(XM)
        YD_channel_attention = self.cbamy(YD)
        # XM = torch.cat((x_m_f1, x_m_f2, x_m_g1, x_m_g2), 1).t()
        #
        # XM = XM.view(1, self.args.view*self.args.gcn_layers, self.args.fm, -1)
        # x_channel_attenttion = self.globalAvgPool_x(XM)


        # YD = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
        #
        # YD = YD.view(1, self.args.view*self.args.gcn_layers, self.args.fd, -1)
        #y_channel_attenttion = self.globalAvgPool_y(YD)


        #
        # x_channel_attenttion = self.globalAvgPool_x(XM)
        # x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        # x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        # x_channel_attenttion = torch.relu(x_channel_attenttion)
        # x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        # x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        # x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        # XM_channel_attention = x_channel_attenttion * XM
        #
        # XM_channel_attention = torch.relu(XM_channel_attention)
        #
        # YD = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
        #
        # YD = YD.view(1, self.args.view*self.args.gcn_layers, self.args.fd, -1)
        #
        # y_channel_attenttion = self.globalAvgPool_y(YD)
        # y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        # y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        # y_channel_attenttion = torch.relu(y_channel_attenttion)
        # y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        # y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        # y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        # YD_channel_attention = y_channel_attenttion * YD
        #
        # YD_channel_attention = torch.relu(YD_channel_attention)



        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()



        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()



        # X = torch.cat((x_m_f2, x_m_g2), 1).t().T
        # Y = torch.cat((y_d_f2, y_d_g2), 1).t().T


        return x.mm(y.t()), x, y
        #return x.mm(y.t()), x, y

    # def getMvEmbedding(self, data):
    #     torch.manual_seed(1)
    #     x_m = torch.randn(self.args.miRNA_number, self.args.fm)
    #     x_d = torch.randn(self.args.disease_number, self.args.fd)
    #
    #     x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
    #         data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
    #     x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
    #         data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
    #
    #     x_m_g1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
    #         data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
    #     x_m_g2 = torch.relu(self.gcn_x2_s(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
    #         data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
    #
    #     y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
    #         data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
    #     y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
    #         data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
    #
    #     y_d_g1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
    #         data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
    #     y_d_g2 = torch.relu(self.gcn_y2_s(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
    #         data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
    #
    #     XM1 = torch.cat((x_m_f1, x_m_f2), 1).t()
    #     XM2 = torch.cat((x_m_g1, x_m_g2), 1).t()
    #     XM3 = torch.cat((x_m_f1, x_m_f2, x_m_g1, x_m_g2), 1).t()
    #
    #     YD1 = torch.cat((y_d_f1, y_d_f2), 1).t()
    #     YD2 = torch.cat((y_d_g1, y_d_g2), 1).t()
    #     YD3 = torch.cat((y_d_f1, y_d_f2, y_d_g1, y_d_g2), 1).t()
    #
    #     XM1 = XM1.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
    #     # x_channel_attenttion = self.globalAvgPool_x(XM)
    #     XM1_channel_attention = self.cbamx(XM1)
    #
    #     YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
    #     # y_channel_attenttion = self.globalAvgPool_y(YD)
    #     YD_channel_attention = self.cbamy(YD)
    #
    #     x = self.cnn_x(XM_channel_attention)
    #     x = x.view(self.args.out_channels, self.args.miRNA_number).t()
    #
    #     y = self.cnn_y(YD_channel_attention)
    #     y = y.view(self.args.out_channels, self.args.disease_number).t()
    #
    #     return x,y