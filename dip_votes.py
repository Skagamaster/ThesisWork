import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

votes = [[1, 2, 5, 4, 3],
         [3, 2, 4, 5, 1],
         [5, 1, 3, 4, 2],
         [1, 2, 4, 3, 5],
         [4, 1, 2, 3, 5],
         [5, 1, 2, 3, 4],
         [5, 4, 3, 1, 2],
         [2, 1, 4, 5, 3],
         [1, 2, 3, 5, 4],
         [3, 2, 4, 1, 5],
         [5, 4, 3, 2, 1],
         [5, 1, 2, 4, 3],
         [5, 1, 3, 4, 2],
         [5, 3, 2, 1, 4],
         [5, 2, 4, 3, 1],
         [5, 1, 3, 4, 2],
         [5, 4, 3, 2, 1],
         [3, 2, 4, 5, 1],
         [2, 4, 3, 5, 1],
         [5, 3, 4, 1, 2],
         [1, 4, 2, 5, 3],
         [3, 4, 2, 5, 1],
         [2, 4, 1, 5, 3]]
votes = np.array(votes).T
votes_run_sum = np.zeros(np.shape(votes))
for i in range(len(votes)):
    x = len(votes[i])
    for j in range(x):
        votes_run_sum[i][j] = votes_run_sum[i][j - 1] + votes[i][j]
vote_hist = np.copy(votes)
vote_hist = np.sort(vote_hist, axis=1)
for i in range(len(vote_hist)):
    vote_hist[i] = vote_hist[i][::-1]
vote_hist = vote_hist.astype('str')
vote_hist = np.where(vote_hist == '5', "first", vote_hist)
vote_hist = np.where(vote_hist == '4', "second", vote_hist)
vote_hist = np.where(vote_hist == '3', "third", vote_hist)
vote_hist = np.where(vote_hist == '2', "fourth", vote_hist)
vote_hist = np.where(vote_hist == '1', "fifth", vote_hist)
order = ['first', 'second', 'third', 'fourth', 'fifth']

plt.figure(figsize=(9, 5))
names = ["Mandarin Orange", "Sulphur Yellow", "Grabber Blue",
         "Nardo Grey", "Wicked Wine"]
print(names)
print(votes_run_sum.T[-1])
colors = ["orange", "gold", "cornflowerblue", "slategrey", "#C20078"]
for i in range(len(votes_run_sum)):
    plt.plot(votes_run_sum[i], color=colors[i], label=names[i]+": "+str(votes_run_sum.T[-1][i]), lw=5, alpha=0.8)
plt.legend()
plt.ylabel("Total Votes", fontsize=20)
plt.xlabel("Vote #", fontsize=20)
plt.title("Running Vote Tally", fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.hist(vote_hist.T, color=colors, stacked=True)
plt.ylabel("Number of Votes", fontsize=20)
plt.xlabel("Place", fontsize=20)
plt.title("Vote Place Totals", fontsize=30)
plt.tight_layout()
plt.show()
