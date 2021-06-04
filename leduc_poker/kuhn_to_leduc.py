import numpy as np
np.set_printoptions(suppress=True)
import random
import argparse
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
torch.set_printoptions(sci_mode=False)
import pickle
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


from environments.leduc_poker import LeducPoker, LeducPop, calc_ev, get_exploitability, leduc_reset, LEDUC_CARDS, LEDUC_ACTIONS
from utils.meta_solver import meta_solver_small, meta_solver_large, fictitious_play

parser = argparse.ArgumentParser(description='Leduc Generalisation Test')
parser.add_argument('--num_test_games', type=int, default=5)
parser.add_argument('--train_iters', type=int, default=40)
parser.add_argument('--br_type', type=str, default='approx_br_rand')
args = parser.parse_args()

def plot_error(data, label=''):
    alpha = .4
    data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data), axis=0)
    plt.plot(data_mean, label=label)
    plt.fill_between([i for i in range(data_mean.size)],
                        np.squeeze(data_mean - error_bars),
                        np.squeeze(data_mean + error_bars), alpha=alpha)

def psro(leduc_pop, train_iters, model=None, br_type='exact'):

  if model == None:
        print('using nash')
        exps = []
        for _ in tqdm(range(train_iters)):
            payoff = leduc_pop.get_metagame()
            meta_nash, _ = fictitious_play(payoffs=payoff, iters=1000)
            agg_agent = leduc_pop.agg_agents(meta_nash[-1])
            exp, br_agent_probs = get_exploitability(agg_agent, skip_illegal=True, br_type=br_type)
            exps.append(exp)
            br_vagent = leduc_pop.agent2vec(br_agent_probs)
            leduc_pop.add_new(np.random.randint(1e5))
            leduc_pop.update(br_vagent, -1, 1)
        
        return exps

  else:
        print('using model')
        exps = []
        for _ in tqdm(range(train_iters)):
            payoff = torch.Tensor(leduc_pop.get_metagame())
            payoff = payoff[None,None,]
            meta_nash = model(payoff)[0].detach().numpy()
            agg_agent = leduc_pop.agg_agents(meta_nash)
            exp, br_agent_probs = get_exploitability(agg_agent, skip_illegal=True, br_type=br_type)
            exps.append(exp)
            br_vagent = leduc_pop.agent2vec(br_agent_probs)
            leduc_pop.add_new(np.random.randint(1e5))
            leduc_pop.update(br_vagent, -1, 1)

        return exps

def run(num_test, psro_iters, br_type):
    seed = np.random.randint(10000)

    for i in range(1):
        model_exps = []
        envs_list = [LeducPoker() for _ in range(num_test)]
        pop_list = [LeducPop(envs_list[i], seed=(seed + i), skip_illegal=True) for i in range(num_test)]
        for k in range(num_test):
            model = meta_solver_small()
            if br_type == 'exact_br':
                with open(f'model_exact_br.pth', 'rb') as f:
                    model.load_state_dict(torch.load(f)) 
            elif br_type == 'approx_br_rand':
                with open(f'model_approx_br_rand.pt', 'rb') as f:
                    model.load_state_dict(torch.load(f))    
            model.eval()
            br_type = br_type
            exp = psro(pop_list[k], psro_iters, model, br_type)  
            model_exps.append(exp)
        d = {'exploitabilities': model_exps}   
        pickle.dump(d, open(os.path.join('results', f'leduc_{br_type}_model_data.p'), 'wb'))
    
    nash_exps = []
    envs_list = [LeducPoker() for _ in range(num_test)]
    pop_list = [LeducPop(envs_list[i], seed=(seed + i), skip_illegal=True) for i in range(num_test)]
    for k in range(num_test):
        exp = psro(pop_list[k], psro_iters, None, br_type=br_type) 
        nash_exps.append(exp)
    d = {'exploitabilities': nash_exps}   
    pickle.dump(d, open(os.path.join('results', f'leduc_{br_type}_nash_data.p'), 'wb'))

    uniform_exps = []
    envs_list = [LeducPoker() for _ in range(num_test)]
    pop_list = [LeducPop(envs_list[i], seed=(seed + i), skip_illegal=True) for i in range(num_test)]
    for k in range(num_test):
        model = meta_solver_small()
        model.eval()
        exp = psro(pop_list[k], psro_iters, model, br_type=br_type) 
        uniform_exps.append(exp)
    d = {'exploitabilities': model_exps}   
    pickle.dump(d, open(os.path.join('results', f'leduc_{br_type}_uniform_data.p'), 'wb'))

    fig = plt.figure(figsize=(8,6))
    plot_error(model_exps, label='Auto-PSRO')
    plot_error(uniform_exps, label='Uniform')
    plot_error(nash_exps, label='Nash')
    plt.legend(loc="upper right")
    plt.title(f'Kuhn Poker {br_type}')
    plt.savefig(os.path.join('results', f'leduc_{br_type}.pdf'))
    plt.close()

if __name__ =="__main__":
    run(args.num_test_games, args.train_iters, args.br_type)