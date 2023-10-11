library(countrycode)
library(ggplot2)
library(cowplot)
library(reshape2)
library(geojsonR)

mean_year_in_school_df = read.csv('data/mean_years_in_school_women_15_to_24_years.csv')
happiness_score_df = read.csv('data/hapiscore_whr.csv')

mean_year_columns_to_drop <- c("X1970","X1971","X1972","X1973","X1974","X1975","X1976","X1977","X1978","X1979", "X1980", "X1981", 
          "X1982","X1983","X1984","X1985","X1986","X1987","X1988","X1989","X1990","X1991", "X1992","X1992",
         "X1993","X1994","X1995","X1996","X1997","X1998","X1999","X2000","X2001","X2002","X2003","X2004")

happines_columns_to_drop <- c("X2016","X2017","X2018","X2019","X2020","X2021","X2022")


years_considered = c("X2005","X2006","X2007","X2008","X2009","X2010","X2011","X2012","X2013","X2014","X2015")

mean_year_in_school_df = mean_year_in_school_df[,!(names(mean_year_in_school_df) %in% mean_year_columns_to_drop)]
happiness_score_df = happiness_score_df[,!(names(happiness_score_df) %in% happines_columns_to_drop)]

head(mean_year_in_school_df)

head(happiness_score_df)

summary(mean_year_in_school_df)

summary(happiness_score_df)

mean_value_imputation <- function(df){
    df2 <- df                                              # Duplicate data frame
    for(i in 1:ncol(df)) {                                   # Replace NA in all columns
      df2[ , i][is.na(df2[ , i])] <- mean(df2[ , i], na.rm = TRUE)
    } 
    df2
}

print("Mean years in school women 15 -> 24 missing syears_consideredamples count")
print(sum(is.na(mean_year_in_school_df)))
print("Happiness score missing samples count")
print(sum(is.na(happiness_score_df)))

happiness_score_imputated_df = mean_value_imputation(happiness_score_df)

print("Happiness score missing samples count")
print(sum(is.na(happiness_score_imputated_df)))

assign_continent <- function(x,output) {
    continent = countrycode(x[1], origin = 'country.name', destination = 'continent')
    continent
}

mean_year_in_school_df$continent = apply(mean_year_in_school_df, 1, assign_continent)
happiness_score_imputated_df$continent = apply(happiness_score_imputated_df,1 , assign_continent)

# Figure scaling
fig <- function(width, heigth){
 options(repr.plot.width = width, repr.plot.height = heigth)
 }

fig(12, 6)
data_long <- melt(mean_year_in_school_df[, c(years_considered, "continent")], id.vars = "continent")
ggplot(data_long, aes(x = variable, y = value, fill = continent)) +
  geom_bar(stat = "identity", width=0.7, position=position_dodge(width=0.8)) +
  labs(x = "Year", y = "Mean years in school of women aged 15 to 24", 
       title = "Mean years in school of women aged 15 to 24 in various continents over the years") +
  theme_minimal()

head(happiness_score_imputated_df)

fig(12, 6)
data_long <- melt(happiness_score_imputated_df[, c(years_considered, "continent")], id.vars = "continent")
ggplot(data_long, aes(x = variable, y = value, fill = continent)) +
  geom_bar(stat = "identity", width=0.7, position=position_dodge(width=0.8)) +
  labs(x = "Year", y = "Happiness Score", 
       title = "Happiness score in various continents over the years from WHR") +
  theme_minimal()

head(mean_year_in_school_df)

mean_year_in_school_africa_df <- mean_year_in_school_df[mean_year_in_school_df$continent == 'Africa', ]
happiness_score_imputated_africa_df <- happiness_score_imputated_df[happiness_score_imputated_df$continent == 'Africa', ]

x = colMeans(mean_year_in_school_africa_df[years_considered])
y = colMeans(happiness_score_imputated_africa_df[years_considered])

fig(12, 6)
ggplot(data.frame(x,y),aes(x=x, y=y)) + geom_point() + labs(x = "Mean years in school of women aged 15 to 24 in various continents over the years", y = "Africa happiness Score", 
       title = "Happiness score in Africa vs Mean years in school of African women aged 15 to 24")

assign_country_code <- function(x,output) {
    code = countrycode(x[1], origin = 'country.name', destination = 'iso3c')
    code
}
  

africa_geojson = FROM_GeoJson(url_file_string = "data/africa_adm0.geojson")

mean_year_in_school_africa_df$country_code = apply(mean_year_in_school_africa_df, 1, assign_country_code)
happiness_score_imputated_africa_df$country_code = apply(happiness_score_imputated_africa_df, 1, assign_country_code)

## None of the libraries could install :(
