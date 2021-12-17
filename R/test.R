library("sas7bdat")
library("corrplot")
library("gtsummary")
library("tidyverse")
library("nnet")
library("lubridate")
library("maps")
library("geosphere")


data = readr::read_csv("03-08.csv")

# select variables
data = data %>% 
  filter(Year == "2008") %>%
  mutate(DepDelayC = cut(DepDelay, breaks=c(-Inf, 5, 30, Inf), 
                         labels=c(0, 1, 2)), 
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
  filter(n > 3650) %>% 
  pull(OD)
data1 = data %>% filter(OD %in% OD_10)

data1 = data1 %>%
  mutate(
    Deptime = strptime(
      paste(Year, Month, DayofMonth, 
            str_sub(CRSDepTime, end=-3), 
            str_sub(CRSDepTime, start=-2), sep="/"), 
      "%Y/%m/%d/%H/%M")
  )

# Visualization: Draw map
airports = read.csv(
  "http://datasets.flowingdata.com/tuts/maparcs/airports.csv", header=TRUE)
airports = airports %>%
  select(iata, lat, long)
data1 = data1 %>% 
  left_join(airports, by=c("Origin"="iata")) %>% 
  rename(c("lat_ori"="lat", "long_ori"="long")) %>%
  left_join(airports, by=c("Dest"="iata")) %>% 
  rename(c("lat_dest"="lat", "long_dest"="long"))
map("state")
n = nrow(data1)
for (i in 1:n){
  inter <- gcIntermediate(
    c(data1[i,]$lat_ori, data1[i,]$long_ori), 
    c(data1[i,]$lat_dest, data1[i,]$long_dest), 
    n=300, addStartEnd=TRUE)
  lines(inter, col="black", lwd=0.8)
}




a = interval(ymd_hms("2008-01-01 1:00:00"), ymd_hms("2008-12-31 1:00:00"))
b = ymd_hms("2008-01-01 1:00:00") + hours(0:2^16)
c = b[b %within% a]
table = data.frame(time=c, start=)

data %>%
  select(Year, Month, DayofMonth, DayOfWeek, CRSDepTime, UniqueCarrier, Origin, Dest, depdelayC) %>%
  tbl_summary(by = depdelayC,
              missing = "no",
              label = list(),
              type = list(Month ~ 'categorical',
                          DayofMonth~ 'categorical',
                          CRSDepTime ~ 'continuous'),
              statistic = list(all_continuous() ~ "{mean} ({sd})",
                               all_categorical() ~ "{n} ({p}%)"),
  ) %>%
  bold_labels

# split train/test data set
set.seed(1)
sub = sample(1:nrow(data),round(nrow(data)*2/3))
data_train = data[sub,]
data_test = data[-sub,]

# logistic regression
m1 = nnet::multinom(DepDelayC ~ Year+Month+DayofMonth+DayofWeek+CRSDepTime+
                      UniqueCarrier+Origin+Dest+Distance, MaxNWts=12000
                    , data=data_train)
summary(m1)
predict(m1)