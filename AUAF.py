#!/usr/bin/env python

import sys
from random import random
from copy import copy
import numpy as np
import datetime

class Supply:
    def __init__(self, supply_file):
        self.idx = {}
        self.id = {}
        self.pair = {}
        self.req_num = []
        self.satisfy_demand = {}
        self.cast_num = []
        self.supply_num = 0
        self.supply_sum = 0
        self.req_sum = 0
        with open(supply_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                i, s, ad_info = line.split('\t')
                self.id[self.supply_num] = i
                self.idx[i] = self.supply_num
                self.pair[i] = int(float(s))
                self.supply_sum += int(float(s))
                self.req_num.append(int(float(s)))
                self.cast_num.append(0)
                self.req_sum += int(float(s))
                self.supply_num += 1
        f.close()
        self.request_num_mat = np.array(self.req_num, dtype=int).reshape(self.supply_num, 1)
        self.cast_num_mat = np.array(self.cast_num, dtype=int).reshape(self.supply_num, 1)

    def get_supply(self, i):
        return self.pair[i]

    def get_supply_id(self, idx):
        return self.id[idx]

    def get_supply_idx(self, i):
        return self.idx[i]

    def get_satisfy_demand(self, i):
        return self.satisfy_demand[i]

    def get_all_i(self):
        return self.pair.keys()

    def get_req_num(self, i):
        return self.req_num[self.idx[i]]

    def get_cast_num(self, i):
        return self.cast_num[self.idx[i]]

    def set_cast_num(self, i, k):
        self.cast_num_mat[self.idx[i]] = k

    def get_req_sum(self):
        return self.req_sum

    def get_supply_num(self):
        return self.supply_num

    def get_supply_sum(self):
        return self.supply_sum


class Demand:
    def __init__(self, demand_file):
        self.demand = []
        self.id = {}
        self.idx = {}
        self.lambda_val = []
        self.weight_val = []
        self.priority = []
        self.target_supply = {}
        self.demand_num = 0
        self.demand_sum = 0
        with open(demand_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                j, d, priority, weight, lambda_ = line.split('\t')
                self.idx[j] = self.demand_num
                self.id[self.demand_num] = j
                self.demand.append(int(float(d))+1)
                self.priority.append(int(float(priority)))
                self.lambda_val.append(float(lambda_))
                self.weight_val.append(float(weight))
                self.demand_num += 1
                self.demand_sum += int(float(d)+1)
        f.close()

        self.demand_mat = np.array(self.demand, dtype=np.float64).reshape(1, self.demand_num)
        self.v_mat = np.array(self.priority, dtype=int).reshape(1, self.demand_num)
        self.lambda_mat = np.array(self.lambda_val, dtype=np.float64).reshape(1, self.demand_num)
        self.weight_mat = np.array(self.weight_val, dtype=np.float64).reshape(1, self.demand_num)

    def _set_supply_satisfy_demand(self, supply):
        for (j, ii) in self.target_supply.items():
            for i in ii:
                if i not in supply.satisfy_demand:
                    supply.satisfy_demand[i] = []
                supply.satisfy_demand[i].append(j)

    def get_demand_id(self, idx):
        return self.id[idx]

    def get_demand_idx(self, j):
        return self.idx[j]

    def get_demand(self, j):
        return self.demand[self.idx[j]]

    def get_target_supply(self, j):
        return self.target_supply[j]

    def get_all_j(self):
        return self.idx.keys()

    def get_v(self, j):
        return self.priority[self.idx[j]]

    def get_lambda(self, j):
        return self.lambda_val[self.idx[j]]

    def get_weight(self, j):
        return self.weight_val[self.idx[j]]

    def get_demand_num(self):
        return self.demand_num

    def get_demand_sum(self):
        return self.demand_sum


class Edge:
    def __init__(self, supply_file, demand, supply):
        self.edge = {}
        supply_num = supply.get_supply_num()
        demand_num = demand.get_demand_num()
        self.gamma = np.zeros((supply_num, demand_num), dtype=int)
        self.ctr = np.zeros((supply_num, demand_num), dtype=np.float64)
        self.target_si = {}
        self.target_req = {}
        self.edge_num = 0
        cnt = 0
        self.same_num = 0
        self.target_num = 0
        with open(supply_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                i, s, ad_info = line.split('\t')
                if i not in supply.satisfy_demand:
                    supply.satisfy_demand[i] = []
                ad_infos = ad_info.split(';')
                cast_cnt = 0
                for demand_ctr_list in ad_infos:
                    d, ctr = demand_ctr_list.split(',')
                    if d not in demand.get_all_j():
                        continue
                    supply.satisfy_demand[i].append(d)
                    if d not in demand.target_supply:
                        demand.target_supply[d] = []
                    demand.target_supply[d].append(i)
                    self.ctr[supply.get_supply_idx(i), demand.get_demand_idx(d)] = float(ctr)
                    self.gamma[supply.get_supply_idx(i), demand.get_demand_idx(d)] = 1
                    cast_cnt += 1
                supply.set_cast_num(i, cast_cnt)
        f.close()
        for j in demand.get_all_j():
            for i in demand.get_target_supply(j):
                self.edge[str(j)+'_'+str(i)] = cnt
                self.target_si[str(j) + '_' + str(i)] = supply.get_supply(i)
                self.target_req[str(j) + '_' + str(i)] = supply.get_supply(i)
                cnt += 1
        self.edge_num = cnt

    def get_edge_idx(self, j_i):
        return self.edge[j_i]

    def get_gamma(self, idx_i, idx_j):
        return self.gamma[idx_i, idx_j]

    def get_edge_num(self):
        return self.edge_num

    def get_edge_target_si(self, j_i):
        return self.target_si[j_i]

    def get_edge_target_req(self, j_i):
        return self.target_req[j_i]

    def get_edge_ctr(self, idx_i, idx_j):
        return self.ctr[idx_i, idx_j]

    def get_target_num(self):
        return self.target_num

    def get_same_target_num(self):
        return self.same_num


class AUAF:
    def __init__(self, supply, demand, edge):
        self.supply = supply
        self.demand = demand
        self.edge = edge
        self.gamma = edge.gamma
        self.error = 1e-10
        self.alpha_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.beta_i = np.zeros((supply.get_supply_num(), 1), dtype=np.float64)
        self.theta_ij = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.x_ij = np.zeros((supply.get_supply_num(), demand.get_demand_num()), dtype=np.float64)
        self.grad_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.si_remained_ratio = np.ones((supply.get_supply_num(), 1), dtype=np.float64)
        self.output_loss = []
        self.output_allocated = []
        self.output_rate = []
        self.loss = {}
        self.click = {}
        self.allocate = {}
        self.last_loss = 0
        self.current_loss = 0
        self.current_allocated = 0
        self.current_click = 0
        self.it = 0
        self.solve = Solve()
        self.pv_rate_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.demand_ctr_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.demand_click_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)
        self.demand_allocated_j = np.zeros((1, demand.get_demand_num()), dtype=np.float64)

    def initialize(self):
        self.alpha_j = self.demand.weight_mat + self.demand.lambda_mat * (self.edge.ctr.max(axis=0).reshape((1, self.demand.get_demand_num())))
        self.theta_ij = self.demand.demand_mat / ((self.supply.request_num_mat * self.gamma).sum(axis=0).reshape(1, self.demand.get_demand_num()) + 0.0001)
        self.calculate_x_ij()
        self.calculate_grad_j()

    def train(self, iter_num):
        iter = iter_num
        while iter > 0:
            self.it = iter_num - iter
            self.update_alpha_j()
            self.calculate_beta_i()
            self.calculate_x_ij()
            self.calculate_grad_j()
            self.calculate_obj()
            print str(iter-1) + 'iters left, current_loss = ' + str(self.current_loss) \
                  + ', last_loss = ' + str(self.last_loss) + ', obj_gain = ' + str(self.current_loss - self.last_loss) \
                  + ', current_allocated = ' + str(self.current_allocated) \
                  + ', current_click = ' + str(self.current_click)
            self.loss[iter_num - iter] = self.current_loss
            self.click[iter_num-iter] = self.current_click
            self.allocate[iter_num-iter] = self.current_allocated
            self.last_loss = self.current_loss
            iter -= 1

    def calculate_x_ij(self):
        #x_ij = max(0, theta * (w_j + (1 + lambda_j * ctr - alpha - beta) / vj))
        self.x_ij = np.minimum(1, np.maximum(0,
                                  self.theta_ij * (1 + (self.demand.weight_mat + self.demand.lambda_mat * self.edge.ctr - self.alpha_j - self.beta_i)
                                                   / self.demand.v_mat))) * self.gamma

    def calculate_beta_i(self):
        a1 = self.alpha_j - self.demand.lambda_mat * self.edge.ctr - self.demand.weight_mat - self.demand.v_mat
        a2 = self.alpha_j - self.demand.lambda_mat * self.edge.ctr - self.demand.weight_mat + self.demand.v_mat * (1.0 / self.theta_ij - 1)
        b = self.theta_ij / self.demand.v_mat
        beta_array = []
        for i in range(self.supply.get_supply_num()):
            coef = []
            cntt = 0
            total_upper_bound = 0.0
            beta_i = self.beta_i[i]
            for j in range(self.demand.get_demand_num()):
                if self.gamma[i, j] <= 0 or b[0, j] == 0:
                    continue
                coef.append((a1[i, j], 0, b[0, j]))
                coef.append((a2[i, j], 1, b[0, j]))
                if a1[i, j] > a2[i, j] + self.error:
                    cntt += 1
                total_upper_bound += 1.0
            if cntt > 0:
                print 'coef length = ' + str(len(coef)) + ', error cnt = ' + str(cntt)
                print coef
            if total_upper_bound < 1.0 + self.error:
                beta_i = 0.0
            else:
                result = self.solve.max(coef, 1.0)
                if result == 10000 or result == 20000:
                    beta_i = 0.0
                else:
                    if result > 0:
                        beta_i = 0.0
                    else:
                        if abs(result + self.beta_i[i]) < self.error:
                            beta_i = self.beta_i[i]
                        else:
                            beta_i = - 1.0 * result
            beta_array.append(beta_i)
        self.beta_i = np.array(beta_array, dtype=np.float32).reshape(self.supply.get_supply_num(), 1)

    def update_beta_i(self):
        #beta_i = max(0, beta_i - (sum_j(xij) - 1))
        self.beta_i = np.maximum(0, self.beta_i + (self.x_ij.sum(axis=1).reshape(self.supply.get_supply_num(),1) - 1))

    def update_alpha_j(self):
        #alpha_j = max(0, alpha_j - vj * (1 - grad_j / dj))
        self.alpha_j = np.maximum(0, self.alpha_j - self.demand.v_mat * (1 - self.grad_j/self.demand.demand_mat))

    def calculate_grad_j(self):
        #grad_j = sum_i(x_ij * s_i)
        self.grad_j = (self.x_ij * self.supply.request_num_mat).sum(axis=0).reshape(1, self.demand.get_demand_num())

    def calculate_obj(self):
        self.demand_click_j = (self.supply.request_num_mat * self.edge.ctr * self.x_ij * self.demand.lambda_mat).sum(axis=0)
        self.current_click = self.demand_click_j.sum()
        self.demand_allocated_j = np.minimum(self.grad_j, self.demand.demand_mat)
        self.current_allocated = self.demand_allocated_j.sum()
        smooth_loss = ((self.supply.request_num_mat * self.demand.v_mat / self.theta_ij
                        * ((self.x_ij - self.theta_ij) ** 2)) * self.gamma).sum() * 0.5
        print 'smooth_loss = ' + str(smooth_loss)
        x_sum_i = self.x_ij.sum(axis=1).reshape(self.supply.get_supply_num(), 1)
        self.current_loss = smooth_loss - self.current_allocated - self.current_click

    def output(self, iter_num):
        print 'loss:'
        print self.loss
        print 'click:'
        print self.click
        print 'xij'
        print self.x_ij
        f = open("result/AUAF_result_" + "_iter" + str(iter_num) + ".txt", "w")
        for i in range(iter_num):
            f.write("iter=" + str(i+1) + ", loss = " + str(self.loss[i]) + ", allocated = " + str(self.allocate[i]) + ", click = " + str(self.click[i]) + '\n')


class Solve:
    def max(self, coef, y):
        sum = 0.0
        sum_k = 0.0
        coef = sorted(coef, key=lambda t: t[0])
        x0 = coef[0][0]
        for entry in coef:
            sum += (entry[0] - x0) * sum_k
            if sum + 1e-8 >= y:
                if sum_k == 0.0:
                    if sum > y:
                        print 'max error 1'
                        return 10000
                else:
                    return entry[0] - (sum - y) / sum_k
            x0 = entry[0]
            if entry[1] == 0:
                sum_k += entry[2]
            else:
                sum_k -= entry[2]
        print 'max error 2'
        return 20000


def main(iter_num_):
    print 'Start at ' + datetime.datetime.now().strftime('%H:%M:%S')

    supply = Supply('supply_data')
    demand = Demand('demand_data')
    print 'load supply and demand finished'
    edge = Edge('supply_data', demand, supply)

    supply_num = supply.get_supply_num()
    supply_sum = supply.get_supply_sum()
    demand_num = demand.get_demand_num()
    demand_sum = demand.get_demand_sum()
    edge_num = edge.get_edge_num()
    req_sum = supply.get_req_sum()

    print 'supply num: ' + str(supply_num)
    print 'supply sum: ' + str(supply_sum)
    print 'demand num: ' + str(demand_num)
    print 'demand sum: ' + str(demand_sum)
    print 'edge num: ' + str(edge_num)
    print 'request sum: ' + str(req_sum)
    print 'same target edge num: ' + str(edge.get_same_target_num())
    print 'target edge num: ' + str(edge.get_target_num())

    np.seterr(divide='ignore', invalid='ignore')
    # np.seterr(divide='warn')
    auaf = AUAF(supply, demand, edge)
    auaf.initialize()
    print 'train start at ' + datetime.datetime.now().strftime('%H:%M:%S')
    auaf.train(iter_num_)
    print 'train finished at ' + datetime.datetime.now().strftime('%H:%M:%S')
    auaf.output(iter_num_)
    print 'AUAF finished at ' + datetime.datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__' :
    iter_num_ = 100
    print 'iter_num = ' + str(iter_num_)
    main(iter_num_)
