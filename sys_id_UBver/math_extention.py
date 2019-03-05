# -*- coding: utf-8 -*-
# author: ub

'''
ライブラリで拡張できない計算関数群．
'''

import numpy as np


def bc2ic(phi, theta, psi, x, y, z):
    '''
    機体座標系から慣性座標系へ変換する関数.

    Parameters
    ----------
    phi : float64
        ロール角.
    theta : float64
        ピッチ角.
    psi : float64
        ヨー角.
    x : float64
        機体座標 x.
    y : float64
        機体座標 y.
    z : float64
        機体座標 z.

    Returns
    -------
    ic : array-like
        変換後の慣性座標 [x, y, z]
    '''

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    euler = np.array(
        [[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
        [sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, sin_phi*cos_theta],
        [cos_phi*sin_theta*cos_psi+sin_phi*sin_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi, cos_phi*cos_theta]]
    )
    bc = np.array([x, y, z])
    ic = np.dot(euler,bc.transpose())

    return ic


def ic2bc(phi, theta, psi, x, y, z):
    '''
    慣性座標系から機体座標系へ変換する関数.

    Parameters
    ----------
    phi : float64
        ロール角.
    theta : float64
        ピッチ角.
    psi : float64
        ヨー角.
    x : float64
        機体座標 x.
    y : float64
        機体座標 y.
    z : float64
        機体座標 z.

    Returns
    -------
    bc : array-like
        変換後の機体座標 [x, y, z]
    '''

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    euler = np.array(
        [[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
        [sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, sin_phi*cos_theta],
        [cos_phi*sin_theta*cos_psi+sin_phi*sin_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi, cos_phi*cos_theta]]
    )
    ic = np.array([x, y, z])
    bc = np.dot(euler.transpose(),ic.transpose())

    return ic


def both_side_diff(x):
    '''
    与えられたリストの中央差分リストを生成して返す関数.
    差分を取るため，要素数は元より２つ少なくなる．

    Parameters
    ----------
    x : array-like

    Returns
    -------
    diff : array-like
        計算結果.

    Raises
    ------
    ValueError
        要素数が２個かそれ以下のリストが与えられるとエラーが起きる．

    Examples
    --------
    >>> x = np.array([1,2,4,5,8])
    >>> diff = both_side_diff(x)
    >>> print(diff)
    [3,3,4]
    '''

    x = np.array(x)
    size = np.size(x)
    diff_x = np.insert(x, [0,0], [0,0]) # はじめの２つの要素を０にする．
    diff_x = np.delete(diff_x,[size,size+1]) # 最後の２つの要素を削除する．
    diff = x - diff_x
    diff = np.delete(diff, [0,1]) # はじめの２つの要素を削除する．

    return diff


def central_diff(y, dx):
    '''
    中央差分法による微分を行なう関数．

    Parameters
    ----------
    y: array-like
    dx: array-like

    Returns
    -------
    dydx: array-like
        計算結果.
    '''

    dydx = both_side_diff(y) / both_side_diff(dx)
    dydx = np.insert(dydx,0,0) # はじめの要素は０にする．
    dydx = np.append(dydx,dydx[-1]) # 最後の要素を一番うしろに追加する．

    return dydx


def lp_filter(t_const, t_diff, data_size, x):
    '''
    一次遅れ要素を用いたローパスフィルタをかける関数．

    Parameters
    ----------
    t_const: float64
        時定数．
    t_diff: float64
        サンプリング間隔．時間偏差．
    data_size: int
        フィルタ処理したいリストのサイズ．
    x: array-like
        フィルタ処理したいリスト．
    n: int
        フィルタの次数．デフォルトは１次．

    Returns
    -------
    lp_x: array-like
        フィルタ処理後のリスト．

    Raises
    ------
    Array-size Error.
        ３次元以上の配列を入力すると起こるエラー．
    '''

    k = t_diff / t_const
    lp_x = np.zeros_like(x)

    # １次元配列の場合
    if x.ndim == 1:
        lp_x[0] = x[0]

        for i in range(data_size-1):
            lp_x[i+1] = k*x[i+1] + (1-k)*lp_x[i]

    # ２次元配列の場合
    elif x.ndim == 2:
        lp_x[0,:] = x[0,:]

        for i in range(data_size-1):
            lp_x[i+1,:] = k*x[i+1,:] + (1-k)*lp_x[i,:]

    else:
        raise Exception('Array-size Error.')

    return lp_x


def fft_set_amp(nparray,dt,N):
    '''
    N個の時系列データを高速フーリエ変換し，振幅を元のスケールに揃えた状態で返す関数．

    Parameters
    ----------
    nparray: array-like
        元のデータ．ndarray型の一次元配列のみ対応．
    dt: float
        サンプリング間隔．
    N: int
        データ数

    Returns
    -------
    fft_nparray: array-like
        処理後の配列．
    '''

    # 高速フーリエ変換（FFT）
    fft = np.fft.fft(nparray)

    # FFTの複素数結果を絶対変換
    fft_amp = np.abs(fft/(N/2))

    return fft_amp
