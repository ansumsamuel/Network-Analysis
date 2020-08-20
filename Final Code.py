
print ()

import networkx
from operator import itemgetter
import matplotlib.pyplot
import pandas as pd

# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('./amazon-books.csv', index_col=0)
#print(amazonBooks.head())

# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()

# Now let's assume a person is considering buying the following book;
# what else can we recommend to them based on copurchase behavior 
# we've seen from other users?
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'

# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])
    

# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
# (1) YOUR CODE HERE: 
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph.
purchasedAsinEgoGraph = networkx.Graph(networkx.ego_graph(copurchaseGraph, purchasedAsin, radius=1))


# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) YOUR CODE HERE: 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for f, t, e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t,weight=e['weight'])


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = [i for i in purchasedAsinEgoTrimGraph.neighbors(purchasedAsin)]
#print(len(purchasedAsinNeighbors))
#print(purchasedAsinNeighbors)

# Next, let's pick the Top Five book recommendations from among the 
# purchasedAsinNeighbors based on one or more of the following data of the 
# neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
# and ClusteringCoeff
# (4) YOUR CODE HERE: 
#     Note that, given an asin, you can get at the metadata associated with  
#     it using amazonBooks (similar to lines 29-36 above).
#     Now, come up with a composite measure to make Top Five book 
#     recommendations based on one or more of the following metrics associated 
#     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
#     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
#     and include other measures if you like.
#     YOU MUST come up with a composite measure.
#     DO NOT simply make recommendations based on sorting!!!
#     Also, remember to transform the data appropriately using 
#     sklearn preprocessing so the composite measure isn't overwhelmed 
#     by measures which are on a higher scale.



#Creating new DF called NeighbourDF for only those ASIN's present in purchasedAsinNeighbors 
NeighborsDF = amazonBooks.filter(purchasedAsinNeighbors, axis=0)
#print(NeighborsDF)

#Preprocessing Step 1: Extracting only those data which has 'SalesRank' not equal -1
NeighborsDF = NeighborsDF[NeighborsDF['SalesRank'] !=-1] 


#Preprocessing Step 2: Scaling 'SalesRank' column using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
MMSNeighborsDF = pd.DataFrame(mms.fit_transform(NeighborsDF[['SalesRank']]), 
                              columns=['MMSalesRank'],
                              index=NeighborsDF.index)
NeighborsDF = pd.concat([NeighborsDF, MMSNeighborsDF], axis=1)


#Creating Composite Key:

#Criteria 1: Giving Priorities to those ASIN's (Node) having the best SalesRank
# and best Rating Conversion Rate
#Rating Conversion Rate = Total Reviews/(Avg Ratings+1) 
#[Note:Add 1 in den to handle 0 Avg Ratings]

#Criteria 2 :Selecting only those ASIN's from Criteria 1 based on best on degree of centrality

#Final composite score based on criteria 1 and Criteria 2 are as follows:

#Composite Score=(Rating Conversion Rate/(MMSalesRank+1))*Degree of Centrality
#[Note:Add 1 in den to handle 0 MMSalesRank]

#Code below:

NeighborsDF['Composite_Measure']=(((NeighborsDF['TotalReviews']/(NeighborsDF['AvgRating']+1)))/(NeighborsDF['MMSalesRank']+1))*NeighborsDF['DegreeCentrality']

print(NeighborsDF.columns)
# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)
# (5) YOUR CODE HERE: 

#Sorting based on Composite measure:

SortingTop5=NeighborsDF.sort_values('Composite_Measure',ascending=False).head(5) 
#print(SortingTop5)

#getting top5 Asins in list called 'AsinList' 
AsinList = [a for a in SortingTop5.index ]
#print(AsinList)

# Let's first get some metadata associated with this book
print()
print ("Top 5 recommendatons for ASIN",purchasedAsin,"are:")
print()

for AsinNo in AsinList:
    
    print ("ASIN = ", AsinNo) 
    print ("Title = ", SortingTop5.loc[AsinNo,'Title'])
    print ("SalesRank = ", SortingTop5.loc[AsinNo,'SalesRank'])
    print ("TotalReviews = ", SortingTop5.loc[AsinNo,'TotalReviews'])
    print ("AvgRating = ", SortingTop5.loc[AsinNo,'AvgRating'])
    print ("DegreeCentrality = ", SortingTop5.loc[AsinNo,'DegreeCentrality'])
    print ("ClusteringCoeff = ", SortingTop5.loc[AsinNo,'ClusteringCoeff'])
    print ("--------------------------------------------------------------------")



