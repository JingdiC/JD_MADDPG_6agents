import random
import pandas as pd
import matplotlib.pyplot as plt




jingdi_spread6_bw1 = pd.read_csv("/Users/Chris/Desktop/JD_MADDPG_6agents/maddpg/learning_curves/jingdi_bw1_spread_6_600_rewards.csv")

SAR_spread6_bw1 = pd.read_csv("/Users/Chris/Desktop/JD_MADDPG_6agents/maddpg/learning_curves/SAR_6agents_bw1_rewards.csv")



plt.plot(jingdi_spread6_bw1, label = "jingdi_spread6_bw1" )

plt.plot(SAR_spread6_bw1, label = "SAR_spread6_bw1")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.ylim(-400,-300)
plt.title("Total Reward Comparison",fontsize=11)
plt.legend(loc='lower right', fontsize=11)
plt.xlabel('Episode',fontsize=11)
plt.ylabel('Total Reward',fontsize=11)
plt.grid(True)


plt.show()