#Applies CAR algorithm to the data and generates association rules for the given dataset.
#param dataset: Path to the CSV file containing the data.
#Required Libraries
library(arules)
library(arulesViz)
library(Rcpp)

#Converts the data to transactions format, applies the CAR algorithm to generate 
#association rules, and plots the top 10 rules with the highest lift.

# Loading Data
dataset <- read.csv(path to dataset)

# Converting variables to factor type
for (i in colnames(dataset)){
  dataset[, i] <- as.factor(dataset[,i])
}

# Data Summary
str(dataset) 

# Converting data to transactions
data_transactions <- as(dataset, "transactions")

# Transaction Summary
dim(data_transactions)

# Getting Item Labels
itemLabels(data_transactions)

# Filtering Item Labels (extract class label)
cols <- itemLabels(data_transactions)
cols = cols[- index of last item]
cols = cols[- index of item before last]


# Generating Association Rules
label1_rules_rhs <- apriori(data_transactions, 
                            parameter = list(supp=0.1, conf=0.9, 
                                             maxlen=10, 
                                             minlen=2),
                            appearance = list(lhs = c(cols), rhs="Class=1"))

# Sorting Rules by Lift
label1_rules_rhs<-sort(label1_rules_rhs,by="lift")
subrules <- head(label1_rules_rhs, n = 10, by = "lift")

# Plotting Top 10 Rules
plot(subrules,method="graph",measure=c("lift"),control=list(type="itemsets"))



