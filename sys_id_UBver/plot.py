# -*- coding: utf-8 -*-
'''
author: ub
プロット関数群．
'''

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

import const


def plot_CL_compare_model(df1,df2):

    # 必要なデータの取り出し
    id = np.array(df1['id'])
    alpha_deg = np.array(df1['alpha_deg'])
    d_alpha = np.array(df1['d_alpha'])
    Va = np.array(df1['Va'])

    CL_log = np.array(df1['CL_log'])
    CL_non_kv = np.array(df1['CL'])
    CL_non_dalpha = np.array(df2['CL'])

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,CL_log,label="Log data",linewidth="3")
    # plt.scatter(Va,CL_non_kv,label=r"Model without $k_L V_a$")
    plt.scatter(Va,CL_non_dalpha,label=r"Model with $k_L V_a$")
    # plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_L^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()


def plot_CD_compare_model(df1,df2):

    # 必要なデータの取り出し
    alpha_deg = np.array(df1['alpha_deg'])
    d_alpha = np.array(df1['d_alpha'])
    Va = np.array(df1['Va'])

    CD_log = np.array(df1['CD_log'])
    CD_non_kv = np.array(df1['CD'])
    CD_non_dalpha = np.array(df2['CD'])

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,CD_log,label="Log data",linewidth="3")
    # plt.scatter(Va,CD_non_kv,label=r"Model without $k_D V_a$")
    plt.scatter(Va,CD_non_dalpha,label=r"Model with $k_D V_a$")
    # plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_D^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()


def plot_Cm_compare_model(df1,df2):

    # 必要なデータの取り出し
    alpha_deg = np.array(df1['alpha_deg'])
    d_alpha = np.array(df1['d_alpha'])
    Va = np.array(df1['Va'])

    Cm_log = np.array(df1['Cm_log'])
    Cm_non_kv = np.array(df1['Cm'])
    Cm_non_dalpha = np.array(df2['Cm'])

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,Cm_log,label="Log data",linewidth="3")
    # plt.scatter(Va,Cm_non_kv,label=r"Model without $k_m V_a$")
    plt.scatter(Va,Cm_non_dalpha,label=r"Model with $k_m V_a$")
    # plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_m^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()


def plot_CL_compare_CFD(df):

    # Vaでフィルタリング
    df_filt = df.query('4.5 <= Va <= 5.5')

    # 必要なデータの取り出し
    alpha = np.array(df_filt['alpha'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    CL_log = np.array(df_filt['CL_log'])
    CD_0 = np.array(df_filt['CD_0'])[0]
    CD_alpha = np.array(df_filt['CD_alpha'])[0]
    CD_k = np.array(df_filt['CD_k'])[0]

    # 同定結果計算
    CL_if_alpha_0 = CD_0 + CD_alpha*0*(pi/180) + CD_k*(1/5)
    CL_if_alpha_m10 = CD_0 + CD_alpha*(-10)*(pi/180) + CD_k*(1/5)
    CL_if_alpha_m20 = CD_0 + CD_alpha*(-20)*(pi/180) + CD_k*(1/5)

    # CFD結果
    CL_CFD_alpha_0 = 0.7118205556
    CL_CFD_alpha_m10 = -0.1093438351
    CL_CFD_alpha_m20 = -1.012350043

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(alpha_deg,CL_log,label="Log data")
    plt.scatter([0,-10,-20],[CL_if_alpha_0,CL_if_alpha_m10,CL_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.scatter([0,-10,-20],[CL_CFD_alpha_0,CL_CFD_alpha_m10,CL_CFD_alpha_m20],label="CFD",color="r",linewidth="8")
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$\alpha \mathrm{[deg]}$',fontsize='36')
    plt.ylabel(r'$C_L^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_CL_compare_CFD2(df):

    # Vaでフィルタリング
    df_filt = df.query('-5 <= alpha_deg <= 5')
    #df_filt = df

    # 必要なデータの取り出し
    Va = np.array(df_filt['Va'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    CL_log = np.array(df_filt['CL_log'])
    CD_0 = np.array(df_filt['CD_0'])[0]
    CD_alpha = np.array(df_filt['CD_alpha'])[0]
    CD_k = np.array(df_filt['CD_k'])[0]

    # 同定結果計算
    # CL_if_alpha_0 = CD_0 + CD_alpha*0*(pi/180) + CD_k*(1/5)
    # CL_if_alpha_m10 = CD_0 + CD_alpha*(-10)*(pi/180) + CD_k*(1/5)
    # CL_if_alpha_m20 = CD_0 + CD_alpha*(-20)*(pi/180) + CD_k*(1/5)

    # CFD結果
    CL_CFD_Va_2 = 2.170995
    CL_CFD_Va_3 = 0.756969
    CL_CFD_Va_4 = 0.572736
    CL_CFD_Va_5 = 0.295664
    CL_CFD_Va_6 = 0.086552
    CL_CFD_Va_7 = 0.063439

    CL_CFDtotal_Va_2 = 44.96154261
    CL_CFDtotal_Va_3 = 19.62610848
    CL_CFDtotal_Va_4 = 11.23839988
    CL_CFDtotal_Va_5 = 7.164452988
    CL_CFDtotal_Va_6 = 4.791634273
    CL_CFDtotal_Va_7 = 3.532731216

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,CL_log,label="Log data")
    # plt.scatter([0,-10,-20],[CL_if_alpha_0,CL_if_alpha_m10,CL_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.plot([2,3,4,5,6,7],[CL_CFD_Va_2,CL_CFD_Va_3,CL_CFD_Va_4,CL_CFD_Va_5,CL_CFD_Va_6,CL_CFD_Va_7],label="CFD",color="r",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[CL_CFDtotal_Va_2-CL_CFD_Va_2,CL_CFDtotal_Va_3-CL_CFD_Va_3,
    #                         CL_CFDtotal_Va_4-CL_CFD_Va_4,CL_CFDtotal_Va_5-CL_CFD_Va_5,
    #                         CL_CFDtotal_Va_6-CL_CFD_Va_6,CL_CFDtotal_Va_7-CL_CFD_Va_7],label="CFD(rotor)",color="g",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[CL_CFDtotal_Va_2,CL_CFDtotal_Va_3,CL_CFDtotal_Va_4,CL_CFDtotal_Va_5,CL_CFDtotal_Va_6,CL_CFDtotal_Va_7],label="CFD(total)",color="b",marker="o",markersize=12)
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_L$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_CD_compare_CFD(df):

    # Vaでフィルタリング
    df_filt = df.query('4.5 <= Va <= 5.5')

    # 必要なデータの取り出し
    alpha = np.array(df_filt['alpha'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    CD_log = np.array(df_filt['CD_log'])
    CD_0 = np.array(df_filt['CD_0'])[0]
    CD_alpha = np.array(df_filt['CD_alpha'])[0]
    CD_k = np.array(df_filt['CD_k'])[0]

    # 同定結果計算
    CD_if_alpha_0 = CD_0 + CD_alpha*0*(pi/180) + CD_k*(1/5)
    CD_if_alpha_m10 = CD_0 + CD_alpha*(-10)*(pi/180) + CD_k*(1/5)
    CD_if_alpha_m20 = CD_0 + CD_alpha*(-20)*(pi/180) + CD_k*(1/5)

    # CFD結果
    CD_CFD_alpha_0 = 1.50520668
    CD_CFD_alpha_m10 = 1.462560459
    CD_CFD_alpha_m20 = 1.547267742

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(alpha_deg,CD_log,label="Log data")
    plt.scatter([0,-10,-20],[CD_if_alpha_0,CD_if_alpha_m10,CD_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.scatter([0,-10,-20],[CD_CFD_alpha_0,CD_CFD_alpha_m10,CD_CFD_alpha_m20],label="CFD",color="r",linewidth="8")
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$\alpha \mathrm{[deg]}$',fontsize='36')
    plt.ylabel(r'$C_D^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_CD_compare_CFD2(df):

    # Vaでフィルタリング
    df_filt = df.query('-5 <= alpha_deg <= 5')
    #df_filt = df

    # 必要なデータの取り出し
    Va = np.array(df_filt['Va'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    CD_log = np.array(df_filt['CD_log'])
    CD_0 = np.array(df_filt['CD_0'])[0]
    CD_alpha = np.array(df_filt['CD_alpha'])[0]
    CD_k = np.array(df_filt['CD_k'])[0]

    # 同定結果計算
    # CD_if_alpha_0 = CD_0 + CD_alpha*0*(pi/180) + CD_k*(1/5)
    # CD_if_alpha_m10 = CD_0 + CD_alpha*(-10)*(pi/180) + CD_k*(1/5)
    # CD_if_alpha_m20 = CD_0 + CD_alpha*(-20)*(pi/180) + CD_k*(1/5)

    # CFD結果
    CD_CFD_Va_2 = 7.025043
    CD_CFD_Va_3 = 3.347775
    CD_CFD_Va_4 = 2.009941
    CD_CFD_Va_5 = 1.294300
    CD_CFD_Va_6 = 0.958428
    CD_CFD_Va_7 = 0.808177

    CD_CFDtotal_Va_2 = 6.78206086
    CD_CFDtotal_Va_3 = 3.271061778
    CD_CFDtotal_Va_4 = 1.982491914
    CD_CFDtotal_Va_5 = 1.272382289
    CD_CFDtotal_Va_6 = 0.946942234
    CD_CFDtotal_Va_7 = 0.804415013

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,CD_log,label="Log data")
    # plt.scatter([0,-10,-20],[CD_if_alpha_0,CD_if_alpha_m10,CD_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.plot([2,3,4,5,6,7],[CD_CFD_Va_2,CD_CFD_Va_3,CD_CFD_Va_4,CD_CFD_Va_5,CD_CFD_Va_6,CD_CFD_Va_7],label="CFD",color="r",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[CD_CFDtotal_Va_2-CD_CFD_Va_2,CD_CFDtotal_Va_3-CD_CFD_Va_3,
    #                         CD_CFDtotal_Va_4-CD_CFD_Va_4,CD_CFDtotal_Va_5-CD_CFD_Va_5,
    #                         CD_CFDtotal_Va_6-CD_CFD_Va_6,CD_CFDtotal_Va_7-CD_CFD_Va_7],label="CFD(rotor)",color="g",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[CD_CFDtotal_Va_2,CD_CFDtotal_Va_3,CD_CFDtotal_Va_4,CD_CFDtotal_Va_5,CD_CFDtotal_Va_6,CD_CFDtotal_Va_7],label="CFD(total)",color="b",marker="o",markersize=12)
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_D$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_Cm_compare_CFD(df):

    # Vaでフィルタリング
    df_filt = df.query('4.5 <= Va <= 5.5')

    # 必要なデータの取り出し
    alpha = np.array(df_filt['alpha'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    Cm_log = np.array(df_filt['Cm_log'])
    Cm_0 = np.array(df_filt['Cm_0'])[0]
    Cm_alpha = np.array(df_filt['Cm_alpha'])[0]
    Cm_k = np.array(df_filt['Cm_k'])[0]

    # 同定結果計算
    Cm_if_alpha_0 = Cm_0 + Cm_alpha*0*(pi/180) + Cm_k*(1/5)
    Cm_if_alpha_m10 = Cm_0 + Cm_alpha*(-10)*(pi/180) + Cm_k*(1/5)
    Cm_if_alpha_m20 = Cm_0 + Cm_alpha*(-20)*(pi/180) + Cm_k*(1/5)

    # CFD結果
    Cm_CFD_alpha_0 = 0.9138172684
    Cm_CFD_alpha_m10 = 0.9350591468
    Cm_CFD_alpha_m20 = 0.8684762804

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(alpha_deg,Cm_log,label="Log data")
    plt.scatter([0,-10,-20],[Cm_if_alpha_0,Cm_if_alpha_m10,Cm_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.scatter([0,-10,-20],[Cm_CFD_alpha_0,Cm_CFD_alpha_m10,Cm_CFD_alpha_m20],label="CFD",color="r",linewidth="8")
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$\alpha \mathrm{[deg]}$',fontsize='36')
    plt.ylabel(r'$C_m^{\prime}$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_Cm_compare_CFD2(df):

    # Vaでフィルタリング
    df_filt = df.query('-5 <= alpha_deg <= 5')
    #df_filt = df

    # 必要なデータの取り出し
    Va = np.array(df_filt['Va'])
    alpha_deg = np.array(df_filt['alpha_deg'])

    Cm_log = np.array(df_filt['Cm_log'])
    Cm_0 = np.array(df_filt['Cm_0'])[0]
    Cm_alpha = np.array(df_filt['Cm_alpha'])[0]
    Cm_k = np.array(df_filt['Cm_k'])[0]

    # 同定結果計算
    # Cm_if_alpha_0 = Cm_0 + Cm_alpha*0*(pi/180) + Cm_k*(1/5)
    # Cm_if_alpha_m10 = Cm_0 + Cm_alpha*(-10)*(pi/180) + Cm_k*(1/5)
    # Cm_if_alpha_m20 = Cm_0 + Cm_alpha*(-20)*(pi/180) + Cm_k*(1/5)

    # CFD結果
    Cm_CFD_Va_2 = 3.937342
    Cm_CFD_Va_3 = 2.032709
    Cm_CFD_Va_4 = 1.279216
    Cm_CFD_Va_5 = 0.991272
    Cm_CFD_Va_6 = 0.794949
    Cm_CFD_Va_7 = 0.693344

    Cm_CFDtotal_Va_2 = 5.88423245
    Cm_CFDtotal_Va_3 = 2.804063288
    Cm_CFDtotal_Va_4 = 1.74628091
    Cm_CFDtotal_Va_5 = 1.27766815
    Cm_CFDtotal_Va_6 = 0.957975279
    Cm_CFDtotal_Va_7 = 0.848452033

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(Va,Cm_log,label="Log data")
    # plt.scatter([0,-10,-20],[Cm_if_alpha_0,Cm_if_alpha_m10,Cm_if_alpha_m20],label="Parameter identification",linewidth="8")
    plt.plot([2,3,4,5,6,7],[Cm_CFD_Va_2,Cm_CFD_Va_3,Cm_CFD_Va_4,Cm_CFD_Va_5,Cm_CFD_Va_6,Cm_CFD_Va_7],label="CFD",color="r",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[Cm_CFDtotal_Va_2-Cm_CFD_Va_2,Cm_CFDtotal_Va_3-Cm_CFD_Va_3,
    #                         Cm_CFDtotal_Va_4-Cm_CFD_Va_4,Cm_CFDtotal_Va_5-Cm_CFD_Va_5,
    #                         Cm_CFDtotal_Va_6-Cm_CFD_Va_6,Cm_CFDtotal_Va_7-Cm_CFD_Va_7],label="CFD(rotor)",color="g",marker="o",markersize=12)
    # plt.plot([2,3,4,5,6,7],[Cm_CFDtotal_Va_2,Cm_CFDtotal_Va_3,Cm_CFDtotal_Va_4,Cm_CFDtotal_Va_5,Cm_CFDtotal_Va_6,Cm_CFDtotal_Va_7],label="CFD(total)",color="b",marker="o",markersize=12)
    plt.grid()

    plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    plt.ylabel(r'$C_m$',fontsize='36')
    plt.tight_layout()
    plt.show()

def plot_eigen_abs(list):

    # データの取り出し
    lambda_A_abs = np.abs(list[0])
    data_size = len(lambda_A_abs)
    x = np.arange(data_size)
    eigen1 = lambda_A_abs[:,0]
    eigen2 = lambda_A_abs[:,1]
    eigen3 = lambda_A_abs[:,2]
    eigen4 = lambda_A_abs[:,3]

    # プロット
    plt.figure()
    plt.subplot(111)
    plt.scatter(x,eigen1,label="")
    plt.scatter(x,eigen2,label="")
    plt.scatter(x,eigen3,label="")
    plt.scatter(x,eigen4,label="")
    plt.grid()

    # plt.legend(fontsize='25')
    plt.tick_params(labelsize='28')

    plt.xlabel(r'Data Number',fontsize='36')
    plt.ylabel(r'Absolute eigenvalue',fontsize='36')
    plt.tight_layout()
    plt.show()
