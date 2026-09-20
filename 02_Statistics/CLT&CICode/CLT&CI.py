import pandas as pd
import numpy as np

#random result should be same 
np.random.seed(42)

n_customers =10000
#Monthly spending 
monthly_spending = np.random.normal(
    loc=5000,
    scale=1500,
    size=n_customers
)
# Negative spending avoid करण्यासाठी
monthly_spending = np.maximum(monthly_spending, 500)

df = pd.DataFrame({
    "Customer_ID": range(1, n_customers + 1),
    "Monthly_Spending": monthly_spending
})

print(df.head())


#check sample mean and population mean
#population mean

p_mean = df["Monthly_Spending"].mean()
print(p_mean)

#sample mean 
sample_size = 100
sample = df["Monthly_Spending"].sample(n=sample_size,random_state=42)

s_mean = sample.mean()

print("p_mean",p_mean)
print("s_mean",s_mean)

###To calculate the Central limit therom , Create the   1000 Samples of data and save the 
### the means of the data
sample_means= []

for i in range (1000):
    sample = df["Monthly_Spending"].sample(n=100)
    sample_means.append(sample.mean())
    
sample_means = np.array(sample_means)

print("\nNumber of sample means:", len(sample_means))
print("Mean of sample means:", sample_means.mean())    
    ############  i have checked the mean of population and mean of 1000 sample is close to same,
#################Now we must check the confidance interval of the main 


# to check the confidance interval formula 
# we have to calculate the Mean of sample data 
sample_mean = sample.mean()

print(sample_mean)
#then standard deviation of sample
sample_std = sample.std(ddof=1) # degree of freedom is n as it is the sample data
print(sample_std)
#standard error calculation formula 
standard_error = sample_std/np.sqrt(sample_size)
print(standard_error)#156
#आपण Sample Size च्या वर्गमुळाने का भागतो?
##जसजसा तुमचा Sample Size वाढतो (उदा. १०० ऐवजी ५०० ग्राहक घेतले), तसतसा तुमचा अंदाज जास्त अचूक होतो आणि 
#तुमची 'चूक' (Error) कमी होते. गणिताच्या (Statistics च्या) नियमानुसार ही त्रुटी Sample Size च्या 
#वर्गमुळाच्या (square root) पटीत कमी होते, म्हणून आपण त्याने भागतो.

margin_of_error = 1.96 * standard_error # fixed formula for margin of error

lower_limit = sample_mean - margin_of_error
upper_limit = sample_mean + margin_of_error     
print("margin of err",margin_of_error)
print("lower limit",lower_limit )
print("upper limit",upper_limit )

###############Calculate 100 of confidance intervals
population_mean = df["Monthly_Spending"].mean()
successful_intervals = 0

for i in range(100):

    sample = df["Monthly_Spending"].sample(n=100)

    mean = sample.mean()
    std = sample.std(ddof=1)

    se = std / np.sqrt(100)

    margin = 1.96 * se

    lower = mean - margin
    upper = mean + margin

    if lower <= population_mean <= upper:
        successful_intervals += 1

print("\nPopulation mean captured in:",
      successful_intervals,
      "out of 100 intervals")


# confiance interavl ----> on sample --->
# 1.calculate mean of sample data 
# ,2.standard deviation 
# 3. standard error
# 95 % C Interval

# for CLT calculate sample mean of 1000s of samples in data

      