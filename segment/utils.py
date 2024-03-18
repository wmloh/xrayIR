import numpy as np
import scipy.sparse as sp
import torch
import cv2

from segment.HybridGNet2IGSC import Hybrid


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    # sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


## Adjacency Matrix
def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i - 1] = 1
        sub[i, (i + 1) % N] = 1
    return sub


## Downsampling Matrix
def mOrganD(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N2, N])

    for i in range(0, N2):
        if (2 * i + 1) == N:
            sub[i, 2 * i] = 1
        else:
            sub[i, 2 * i] = 1 / 2
            sub[i, 2 * i + 1] = 1 / 2

    return sub


def mOrganU(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N, N2])

    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i // 2] = 1
        else:
            sub[i, i // 2] = 1 / 2
            sub[i, (i // 2 + 1) % N2] = 1 / 2

    return sub


def genMatrixesLungsHeart():
    RLUNG = 44
    LLUNG = 50
    HEART = 26

    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)

    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))

    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)

    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)

    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART

    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))

    A = np.zeros([p3, p3])

    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3

    AD = np.zeros([p3_, p3_])

    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3

    D = np.zeros([p3_, p3])

    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3

    U = np.zeros([p3, p3_])

    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3

    return A, AD, D, U


def loadModel(weight_path, device):
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f // 2, f // 2, f // 2]
    config['skip_features'] = f

    hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    hybrid.eval()

    return hybrid


def pad_to_square(img):
    h, w = img.shape[:2]

    if h > w:
        padw = (h - w)
        auxw = padw % 2
        img = np.pad(img, ((0, 0), (padw // 2, padw // 2 + auxw)), 'constant')

        padh = 0
        auxh = 0

    else:
        padh = (w - h)
        auxh = padh % 2
        img = np.pad(img, ((padh // 2, padh // 2 + auxh), (0, 0)), 'constant')

        padw = 0
        auxw = 0

    return img, (padh, padw, auxh, auxw)


def preprocess(input_img):
    img, padding = pad_to_square(input_img)

    h, w = img.shape[:2]
    if h != 1024 or w != 1024:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    return img, (h, w, padding)


def removePreprocess(output, info):
    h, w, padding = info

    if h != 1024 or w != 1024:
        output = output * h
    else:
        output = output * 1024

    padh, padw, auxh, auxw = padding

    output[:, 0] = output[:, 0] - padw // 2
    output[:, 1] = output[:, 1] - padh // 2

    return output


def getMasks(landmarks, h, w):
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]

    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    RL_mask = np.zeros([h, w], dtype='uint8')
    LL_mask = np.zeros([h, w], dtype='uint8')
    H_mask = np.zeros([h, w], dtype='uint8')

    RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
    LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
    H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

    return RL_mask, LL_mask, H_mask


def segment(input_img, segmodel, device):
    # input_img = cv2.imread(input_img, 0) / 255.0
    original_shape = input_img.shape[:2]

    img, (h, w, padding) = preprocess(input_img)

    data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = segmodel(data)[0].cpu().numpy().reshape(-1, 2)

    output = removePreprocess(output, (h, w, padding))

    output = output.astype('int')
    RL_mask, LL_mask, H_mask = getMasks(output, original_shape[0], original_shape[1])

    return RL_mask, LL_mask, H_mask
