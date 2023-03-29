import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

# 世界座標下的方框位置，按照aruco編號(X, Y, Z) (left_top, right_top, right_bottom, left_bottom) (id=0, id=1, id=2)
# 座標需要自己去量! 
world_points = 0.1 * np.array([[18., 0, 112],
                               [108, 0, 112],
                               [111, 0, 23],
                               [20, 0, 22],
                               [13, 13, 0],
                               [102, 20, 0],
                               [102, 112, 0],
                               [16, 108, 0],
                               [0, 112, 110],
                               [0, 22, 112],
                               [0, 18, 22],
                               [0, 109, 19]])

# 紀錄像素座標下各角點的位置，按照aruco編號(left_top, right_top, right_bottom, left_bottom) (id=0, id=1, id=2)
corners_pixel = {0: None, 1: None, 2: None}


def Solve_PM(world_points, corners, N):
    """
    :param corners: 特徵點在像素平面下的座標(N, 2) , (u, v)
    :param world_points: 特徵點在世界座標下的位置(N, 3) , (X, Y, Z)
    :param N: 特徵點數量
    :return:
    """

    # 將world_points轉換成齊次座標 (X, Y, Z) -> (X, Y, Z, 1), shape=(N, 4)
    world_points = np.hstack((world_points, np.ones(shape=(world_points.shape[0], 1))))

    # 創建係數矩陣 shape=(2 * N, 12)
    A = np.zeros(shape=(2 * N, 12), dtype=np.float64)

    # 建立下列矩陣:
    #
    # | X1, Y1, Z1, 1, 0, 0, 0, 0, -u1*X1, -u1*Y1, -u1*Z1, -u1 |
    # | 0, 0, 0, 0, X1, Y1, Z1, 1, -v1*X1, -v1*Y1, -v1*Z1, -v1 |
    # | X2, Y2, Z2, 1, 0, 0, 0, 0, -u2*X2, -u2*Y2, -u2*Z2, -u2 |
    # | 0, 0, 0, 0, X2, Y2, Z2, 1, -v2*X2, -v2*Y2, -v2*Z2, -v2 |
    # |                         .                              |
    # |                         .                              |
    # |                         .                              |
    for i in range(N):
        A[2 * i, :4] = world_points[i]
        A[2 * i, 8:] = -corners[i, 0] * world_points[i]
        A[2 * i + 1, 4:8] = world_points[i]
        A[2 * i + 1, 8:] = -corners[i, 1] * world_points[i]

    # 將P矩陣進行svd分解
    U, S, V_T = np.linalg.svd(A)

    # 解為V_T最後一列的所有元素(實際在計算中為V的最後一行，但是因為現在是V_T，所以對應的解是最後一列), shape=(3, 4)
    M = V_T[-1, :].reshape((3, 4))

    # M = K[R|T] = [KR |KT], 只要有辦法分離出K和R，就有辦法取得K、R、T
    # 仔細看後可以發現，其中K為一個上三角矩陣、R為一個正交矩陣，我們可以利用QR分解!
    KR = M[:, :3]

    # 但是如果直接利用QR分解出來的結果為(正交矩陣) @ (上三角矩陣)，所以要再做一些手腳，讓其變成(上三角矩陣) @ (正交矩陣)
    # 定義一個P矩陣。 可以發現他是一個正交矩陣，滿族 P_inv = P.T = P
    # 並且如果有一個矩陣 B = |0 ,a ,b|
    #                      |0, 0, c|
    #                      |0, 0, 0|
    #
    # 則:
    # PB = |0 ,0 ,0|   BP = |b ,a ,0|
    #      |0, 0, c|        |c, 0, 0|
    #      |0, a, b|        |0, 0, 0|
    #
    # 可以發現PA會將A上下顛倒，AP會將A左右相反
    # Step1: 先將A' = P @ A
    # Step2: 將A'.T 進行QR分解得出
    #         A'.T = Q @ R
    # Step3: 將Step1的結果帶入Step2
    #        (P @ A).T = Q @ R
    #        A.T @ P.T = QR
    #        A.T = QRP
    #        A = P.T @ R.T @ Q.T
    #        A = P @ R.T @ Q.T
    #
    # 到這邊可以發現R.T是一個下三角矩陣，這時候就可以利用我們的P矩陣了!
    # 將一個下三角矩陣先左右翻轉再上下顛倒，或是先上下顛倒再左右翻轉，則可以變為上三角矩陣
    # 即 P @ R.T @ P 為上三角矩陣
    # 且 PP = I
    # 所以:
    #       A = (P @ R.T @ P) @ (P @ Q.T)
    #       A = (P @ R.T @ I) @ Q.T
    #
    # 剛好可以使A = (上三角矩陣) @ (正交矩陣)
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    q, r = np.linalg.qr((P @ KR).T)

    # 上三角矩陣
    K = P @ r.T @ P

    # 正交矩陣
    R = P @ q.T

    # 求解平移矩陣T(M的最後一行是KT)。
    # M = [KR | KT], T = K_inv @ KT = K_inv @ M[:, -1]
    t = np.linalg.inv(K) @ M[:, -1]

    # 組合出外部參數矩陣
    E = np.hstack((R, t.reshape((3, 1))))

    return K, E


# 開啟相機
cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# 建立3d圖
fig = plt.figure()
plt.ion()

while True:
    ret, img = cap.read()

    # 畫3d圖
    fig.clf()
    _3d_figure = fig.add_subplot(111, projection='3d')
    _3d_figure.view_init(elev=25., azim=45)
    _3d_figure.set_xlim(0, 50)
    _3d_figure.set_ylim(0, 50)
    _3d_figure.set_zlim(0, 50)
    _3d_figure.set_xlabel("X")
    _3d_figure.set_ylabel("Y")
    _3d_figure.set_zlabel("Z")

    # 畫出世界坐標系下的定位點
    _3d_figure.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='r')

    # 檢查角點
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, label, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 判斷相機是否有拍攝到角點
    if corners:
        label = np.array(label).reshape((-1,))
        corners = np.array(corners).reshape((-1, 4, 2))  # (n, 4, 2)

        # 判斷是否有12個定位點
        if 0 in label and 1 in label and 2 in label:
            for idx, corner in enumerate(corners):
                left_top = (int(corner[0][0]), int(corner[0][1]))
                right_top = (int(corner[1][0]), int(corner[1][1]))
                right_bottom = (int(corner[2][0]), int(corner[2][1]))
                left_bottom = (int(corner[3][0]), int(corner[3][1]))

                # 將讀取到的頂點依照左上、右上、右下、左下的順序儲存
                corners_pixel[label[idx]] = np.array([[left_top],
                                                      [right_top],
                                                      [right_bottom],
                                                      [left_bottom]])

            # 計算得到相機內參、外參
            K, E = Solve_PM(world_points,
                            np.vstack((corners_pixel[0], corners_pixel[1], corners_pixel[2])).reshape((12, 2)),
                            12)

            # -------------------------------------------------------------------------------
            # -----------------------------------計算相機姿態----------------------------------
            # -------------------------------------------------------------------------------
            # 外參矩陣是將世界坐標系的某一點 -> 相機坐標系中的某一點
            # 那反之取其反矩陣，不就變成將相機坐標系中的某一點 -> 世界坐標系的某一點
            E = np.vstack((E, np.array([0, 0, 0, 1])))
            pos = np.linalg.inv(E) @ np.array([[0],
                                               [0],
                                               [0],
                                               [1]])
            pos = pos.reshape((-1,))
            # 畫出相機在世界座標下的位置(這裡X,Y反過來是因為matplotlib的X,Y軸跟我世界座標是相反的)
            _3d_figure.scatter(pos[1], pos[0], pos[2], c='b')

            # 相機坐標系與世界坐標系距離
            print("相機坐標系與世界坐標系距離:", np.linalg.norm(pos))

    plt.pause(0.01)
    # 顯示圖像和遮罩
    cv2.imshow('image', img)

    # 按下Esc鍵退出程式
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
