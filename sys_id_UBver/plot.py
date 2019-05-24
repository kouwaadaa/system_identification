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
    plt.scatter(Va,CL_non_kv,label=r"Model without $k_L V_a$")
    plt.scatter(Va,CL_non_dalpha,label=r"Model with $k_L V_a$")
    plt.legend(fontsize='25')
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
    plt.scatter(Va,CD_non_kv,label=r"Model without $k_D V_a$")
    plt.scatter(Va,CD_non_dalpha,label=r"Model with $k_D V_a$")
    plt.legend(fontsize='25')
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
    plt.scatter(Va,Cm_non_kv,label=r"Model without $k_m V_a$")
    plt.scatter(Va,Cm_non_dalpha,label=r"Model with $k_m V_a$")
    plt.legend(fontsize='25')
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
    CL_0 = np.array(df_filt['CL_0'])[0]
    CL_alpha = np.array(df_filt['CL_alpha'])[0]
    CL_k = np.array(df_filt['CL_k'])[0]

    # 同定結果計算
    CL_if_alpha_0 = CL_0 + CL_alpha*0*(pi/180) + CL_k*(1/5)
    CL_if_alpha_m10 = CL_0 + CL_alpha*(-10)*(pi/180) + CL_k*(1/5)
    CL_if_alpha_m20 = CL_0 + CL_alpha*(-20)*(pi/180) + CL_k*(1/5)

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
