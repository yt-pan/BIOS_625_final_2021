library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(ggpubr)

setwd("")
data<-read.csv('03-08-str.csv',sep=',',header=TRUE)

p1 <- data %>%
  ggplot( aes(x=CRSDepTime, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 24) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p2 <- data %>%
  ggplot( aes(x=CRSArrTime, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 24) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p3 <- data %>%
  ggplot( aes(x=Month, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 12) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p4 <- data %>%
  ggplot( aes(x=DayofMonth, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 31) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p5 <- data %>%
  ggplot( aes(x=DayOfWeek, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 7) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p6 <- data %>%
  ggplot( aes(x=Distance, fill=depdelayC)) +
    geom_histogram(aes(y=..density..), color="#e9ecef", alpha=0.6, position = 'identity', bins = 20) +
    scale_fill_manual(values=c("red", "lightgreen")) +
    theme_ipsum() +
    labs(fill="") +
    theme_bw() +
    theme(axis.title.y = element_blank(),axis.text.y=element_blank()) +
    theme(panel.grid = element_blank()) +
    theme(legend.position ="none")

p = ggarrange(p1, p2, p3, p4, p5, p6,
          labels = c("A", "B", "C", "D", "E", "F"),
          ncol = 3, nrow = 2)

ggsave(p,filename = "p.png",width = 12,height = 9)