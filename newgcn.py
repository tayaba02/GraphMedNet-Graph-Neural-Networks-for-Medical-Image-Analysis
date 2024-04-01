import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import cv2
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def get_graphparams(img):
    AnnMap = np.zeros((100,100),np.uint8)
    xvec = []
    yvec = []
    transformImg=tf.Compose([tf.ToPILImage(), tf.Resize((100,100)), tf.ToTensor()])
    
    img = img.astype(np.uint8)
    imgT = transformImg(img)
    img = []
    img = 255*imgT.T
    img = img.numpy()
    img = (255*(img - np.min(img))/np.ptp(img)).astype(np.uint8)
    
    corners = cv2.goodFeaturesToTrack(img, 50, 0.01, 10)
    
    if corners is not None:
        corners = np.int0(corners)
        count = 0
        for i in corners:
            x, y = i.ravel()
            xvec.append(x)
            yvec.append(y)
    else:
        xvec = np.zeros(8)
        yvec = np.zeros(8)
            
    img_edge = cv2.Canny(img,100,200)
    idx = np.where(img_edge != [0])

    return idx,xvec,yvec
    
def get_imagedata(dirname):
    
    TrainFolder="Data//"
    Listdir=os.listdir(os.path.join(TrainFolder, dirname)) # Create list of Segmentations
    transformImg=tf.Compose([tf.ToPILImage(), tf.Resize((100,100)), tf.ToTensor()])

    idx_edge = []
    xvec = []
    yvec = []

    count = 1
    for l in Listdir[0:1]:
        n1_img = nib.load(os.path.join(TrainFolder, dirname ,l))
        img_vol = n1_img.get_fdata()
        for im in range(img_vol.shape[2]):
            Img = img_vol[:,:,im]
            t1, t2, t3 = get_graphparams(Img)
            idx_edge.append(t1)
            xvec.append(t2)
            yvec.append(t3)
            
    return idx_edge,xvec,yvec

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
def plot_dataset(dataset):
    edges_raw = dataset.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
                'node_size': 30,
                'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()

def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))
    
def testit(data):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]
    correct += pred.eq.sum().item()
    return correct / (len(data.y))

def train(data, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

            train_acc = testit(data)
            return train_acc
            # test_acc = test(data, train=False)

            # train_accuracies.append(train_acc)
    #         test_accuracies.append(test_acc)
    #         print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
    #               format(epoch, loss, train_acc, test_acc))

    # if plot:
    #     plt.plot(train_accuracies, label="Train accuracy")
    #     plt.plot(test_accuracies, label="Validation accuracy")
    #     plt.xlabel("# Epoch")
    #     plt.ylabel("Accuracy")
    #     plt.legend(loc='upper right')
    #     plt.show()

if __name__ == "__main__":
    
    # Create Dataset
    edge_vol = []
    edge_seg = []
    xvec_vol = []
    xvec_seg = []
    yvec_vol = []
    yvec_seg = []

    dir_vol = "COVID-19-CT-Seg_20cases"
    dir_seg = "Lung_and_Infection_Mask"

    edge_vol, xvec_vol, yvec_vol = get_imagedata(dir_vol)
    edge_seg, xvec_seg, yvec_seg = get_imagedata(dir_seg)
    
    # edgex =  
    trainimg_no = 89
    ann_idx = np.zeros((len(edge_seg[trainimg_no][0]),2))
    for i in range(len(edge_seg[trainimg_no][0])):
        ann_idx[i] = [edge_seg[trainimg_no][0][i], edge_seg[trainimg_no][1][i]]
    
    edge_idx = np.zeros((len(edge_vol[trainimg_no][0]),2))
    trgt = np.zeros(len(edge_vol[trainimg_no][0]))
    for i in range(len(edge_vol[trainimg_no][0])):
        edge_idx[i] = [edge_vol[trainimg_no][0][i], edge_vol[trainimg_no][1][i]]        
        for j in range(len(ann_idx)):
            if (edge_idx[i][0] == ann_idx[j][0]) and (edge_idx[i][1] == ann_idx[j][1]):
                trgt[i] = 1
                
    node_idx = np.zeros((len(xvec_vol[trainimg_no]),2))
    for i in range(len(xvec_vol[trainimg_no])):
        node_idx[i] = (xvec_vol[trainimg_no][i], yvec_vol[trainimg_no][i])
    
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
    node_idx = torch.tensor(node_idx, dtype=torch.float)
    trgt = torch.tensor(trgt, dtype=int)
    dataset2 = Data(x=node_idx, edge_index=edge_idx.t().contiguous(), y = trgt)

    # plot_dataset(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = Net(dataset2).to(device)
    data = dataset2.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(data, plot=True)