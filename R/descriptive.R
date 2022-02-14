library(ggplot2)
library(tidyverse)
library(sf)
library(tmap)

caserate <- readr::read_csv("../data/caserate-by-modzcta.csv") %>%
  pivot_longer(cols = 2:184, names_to = "region", values_to = "caserate") %>%
  mutate(week_ending = as.Date(week_ending, format = "%m/%d/%Y"))

caserate_zcta <- caserate %>% filter(!(region %in% c("CASERATE_SI", "CASERATE_QN", "CASERATE_MN", "CASERATE_CITY", "CASERATE_BX", "CASERATE_BK")))
caserate_city <- caserate %>% filter(region == "CASERATE_CITY")


ggplot() +
  theme_classic() +
  geom_line(
    data = caserate_zcta,
    aes(x = week_ending, y = caserate, color = region),
    alpha = 0.2
  ) +
  theme(legend.position = "none") +
  geom_line(
    data = caserate_city,
    aes(x = week_ending, y = caserate),
    color = "darkred",
    size = 1.5
  )


percentpositive <- readr::read_csv("../data/percentpositive-by-modzcta.csv") %>%
  pivot_longer(cols = 2:184, names_to = "region", values_to = "pct_rate") %>%
  mutate(week_ending = as.Date(week_ending, format = "%m/%d/%Y"))

positive_zcta <- percentpositive %>% filter(!(region %in% c("PCTPOS_SI", "PCTPOS_QN", "PCTPOS_MN", "PCTPOS_CITY", "PCTPOS_BX", "PCTPOS_BK")))
positive_city <- percentpositive %>% filter(region == "PCTPOS_CITY")


ggplot() +
  theme_classic() +
  geom_line(
    data = positive_zcta,
    aes(x = week_ending, y = pct_rate, color = region),
    alpha = 0.2
  ) +
  theme(legend.position = "none") +
  geom_line(
    data = positive_city,
    aes(x = week_ending, y = pct_rate),
    color = "darkred",
    size = 1.5
  )

nycZcta <- st_read("../data/nyc_shp/nyu_2451_34509.shp")
caserate_zcta <- caserate_zcta %>% separate(., col = region, into = c("n", "zcta"), sep = "_")
nycZcta <- nycZcta %>%
  left_join(., caserate_zcta, on = "zcta") %>%
  select(-note, -bcode)

tm_shape(nycZcta %>% filter(week_ending == "2021-01-09")) +
  tm_polygons(col = "caserate", legend.hist = TRUE, n = 5, style = "jenks")






nyc_place <- readr::read_csv("../data/NY_places.csv")


nyc_place %>%
  group_by(sg_c__top_category) %>%
  count() %>%
  filter(n > 800)

dept <- tm_shape(nycZcta) + tm_borders() +
  tm_shape(nyc_place %>% filter(sg_c__top_category == "Department Stores") %>% st_as_sf(coords = c("sg_c__longitude", "sg_c__latitude"))) + tm_dots(size = 0.25, col = "red") +
  tm_layout(legend.position = c("left", "top"), title = "Department Stores", title.position = c("left", "top"))

grocery <- tm_shape(nycZcta) + tm_borders() +
  tm_shape(nyc_place %>% filter(sg_c__top_category == "Grocery Stores") %>% st_as_sf(coords = c("sg_c__longitude", "sg_c__latitude"))) + tm_dots(size = 0.1, col = "red") +
  tm_layout(legend.position = c("left", "top"), title = "Grocery Stores", title.position = c("left", "top"))

liqor <- tm_shape(nycZcta) + tm_borders() +
  tm_shape(nyc_place %>% filter(sg_c__top_category == "Drinking Places (Alcoholic Beverages)") %>% st_as_sf(coords = c("sg_c__longitude", "sg_c__latitude"))) + tm_dots(size = 0.1, col = "red") +
  tm_layout(legend.position = c("left", "top"), title = "Liqor Stores", title.position = c("left", "top"))

rest <- tm_shape(nycZcta) + tm_borders() +
  tm_shape(nyc_place %>% filter(sg_c__top_category == "Restaurants and Other Eating Places") %>% st_as_sf(coords = c("sg_c__longitude", "sg_c__latitude"))) + tm_dots(col = "red") +
  tm_layout(legend.position = c("left", "top"), title = "Restaurants", title.position = c("left", "top"))

tmap_arrange(
  dept, grocery, liqor, rest,
  ncol = 2,
  nrow = 2
)


unique(nyc_place$sg_c__top_category)