library("sas7bdat")
library("corrplot")
library("gtsummary")
library("tidyverse")
library("maps")
library("geosphere")
library("usmap")
library("e1071")

data = readr::read_csv("03-08.csv")

# select variables
data = data %>% 
  mutate(DepDelayC = cut(DepDelay, breaks=c(-Inf, 0, Inf), 
                         labels=c(0, 1)), 
         Year = factor(Year), 
         Month = factor(Month), 
         DayofMonth = factor(DayofMonth), 
         DayofWeek = factor(DayOfWeek), 
         UniqueCarrier = factor(UniqueCarrier), 
         FlightNum = factor(FlightNum), 
         Origin = factor(Origin), 
         Dest = factor(Dest),
         OD = paste(Origin, Dest, sep="-")
  )

# select OD pairs with more than 10 flights per day on average
OD_10 = data %>% group_by(OD) %>%
  summarize(n = n()) %>%
  filter(n > 3650*6+1) %>% 
  pull(OD)
data1 = data %>% filter(OD %in% OD_10)

# Visualization: Draw map
# Flights route map with more than 10 flights per day on average 
airports = read.csv(
  "http://datasets.flowingdata.com/tuts/maparcs/airports.csv", 
  header=TRUE) %>%
  select(iata, lat, long)
flight = data1 %>% 
  select(Origin, Dest) %>%
  group_by(Origin, Dest) %>%
  summarize(cnt=n())
maxcnt = max(flight$cnt)
pal <- colorRampPalette(c("#f2f2f2", "black"))
colors <- pal(100)
map("state", col="#f2f2f2", fill=TRUE, bg="white", lwd=0.05)
n = nrow(flight)
for (i in 1:n){
  air1 <- airports[airports$iata == flight[i,]$Origin,]
  air2 <- airports[airports$iata == flight[i,]$Dest,]
  inter <- gcIntermediate(
    c(air1[1,]$long, air1[1,]$lat), 
    c(air2[1,]$long, air2[1,]$lat), n=100, addStartEnd=TRUE)
  colindex <- round( (flight[i,]$cnt / maxcnt) * length(colors) )
  lines(inter, col=colors[colindex], lwd=0.8)
}

# Delayed flights route map with more than 10 flights per day on average 
flight = data1 %>% 
  filter(DepDelay>0) %>% 
  select(Origin, Dest) %>%
  group_by(Origin, Dest) %>%
  summarize(cnt=n())
maxcnt = max(flight$cnt)
pal <- colorRampPalette(c("#f2f2f2", "black"))
colors <- pal(100)
map("state", col="#f2f2f2", fill=TRUE, bg="white", lwd=0.05)
n = nrow(flight)
for (i in 1:n){
  air1 <- airports[airports$iata == flight[i,]$Origin,]
  air2 <- airports[airports$iata == flight[i,]$Dest,]
  inter <- gcIntermediate(
    c(air1[1,]$long, air1[1,]$lat), 
    c(air2[1,]$long, air2[1,]$lat), n=100, addStartEnd=TRUE)
  colindex <- round( (flight[i,]$cnt / maxcnt) * length(colors) )
  lines(inter, col=colors[colindex], lwd=0.8)
}