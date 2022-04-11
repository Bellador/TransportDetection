---
marp: true
author: Maximilian Hartmann
html: true
size: 4:3
theme: uncover
paginate: true
---

<!-- _paginate: false -->
<!-- _color: #101010 -->

    Videos as Datasource for 
    Transportation Mode Detection


![bg](./titlepage.png)

<!-- Start: Presentation on transport mode detection. How dow people get from A to B in urban areas? And how can we measure that? -->
--- 
## Why?

Transport is responsible for 24% of energy-related CO2 emissions 

###### (International Energy Agency; 2018)

.. we need to know how the mobility mix changes.

---
### Frameworks


##### Sustainable Development Goals  ![width:100](./SDG11.jpg)
68% of environment related indicators lack data availability 
###### (UN Environment Programme)

<!-- transport and mobility is central to SDG11 -->

---

### Indicator Requirements

- :black_square_button: spatial resolution
- :black_square_button: temporal resolution
- :black_square_button: feasible in cost and effort
- :black_square_button: reproducible


---

#### :chart_with_upwards_trend: Indicator for urban mobility

# :walking: , :bicyclist: 
###### (Wang et. al. 2020)

# :car: , :truck:

###### (den Braver et al. 2020)

# :station: , :bus:


<!-- non-motorised transport, non-motorised transport, public transport. Also consider motorcyclists -->

--- 

#### Conventional

- count stations
- surveys

<!-- Downsite: stationary, low differentiation between cyclists and others (e.g. e-scooters) -->

![bg right:40%](./cyclist_counter_bern.jpeg)

---

#### :thought_balloon: Complementary
#


- Micro-blogging e.g. Twitter, Weibo 
###### (Yang et. al. 2018)
- Fitness Apps e.g. STRAVA
###### (Georgios 2016)
- Images e.g. Flickr, Google Street View
######  (Domènech et. al. 2020) (Biljecki & Ito 2021)


<!-- geo-tagged, social media as example for UGC -->
<!-- "Street-level imagery became ingrained as an important urban data source" - but Urban analytics dominated by Google Street View -->

---

### :video_camera: City Tour Videos

![width:840](./ugc_video_tracked_modified.png)


--- 

![bg](./ugc_video_tracked_w_streetname.png)

<!-- some CSI number plate zoom in -->

---

![width:870](./workflow_modified.png)

---

## Object relation

# :bicyclist: = :walking: + :bike:

###### (active transportation)

--- 

## :round_pushpin: Georeferencing

- process detected text
- match against OSM street-names gazetteer

###### (Al-Olimat et al., 2017)

<!-- Levensthein distance -->

---

## Results


---

####  :city_sunrise: Geneva
#
#
#
#
#
#
#
#
#
#
#
#
#


![bg](./all_locations_map.jpeg)

---
##### .. over multiple videos
<iframe src="./location_statistics_plot.html" height="600"></iframe>

---
##### Indicator - relative Proxy

<!-- Come back to the indicator that I want to create, this figure shows the potential of it to fingerprint locations and to compare them. 
Explain that the width of the bars correspond to the absolute values -->

![width:850](locations_statistic_relative_selection.png)


---

    Place de la Taconnerie
#
#
#
#
#
#
#
#
#
#
#
#
#

![bg](place_de_la_taconnerie_modified.jpeg)

---

![bg](place_du_port_modified.jpeg)

---

#### Verification on 5min

###

![width:800](./validation_table_modified.jpeg)

<!-- 5min video excerpt from a predestrian zone/transit mall (Fussgängerzone) -->

---

## :bulb: Conclusion

#

- :ballot_box_with_check: spatial resolution (OCR)
- :black_square_button: temporal resolution
- :ballot_box_with_check: feasible in cost and effort
- :ballot_box_with_check: reproducible

<!-- come back to the indicator requirements at the beginning + formulate some 'Take-Home Message' to summarise the talk -->

---
![invert width:100](./github_logo.png)
###### github.com/Bellador/TransportationDetection

![width:300](./github-qrcode-transportdetection.png)

---

### BACKUP SLIDES

---

![bg](./georeference_map_modified.png)

---

##### single video - aggregated
<iframe src="./figure_across_video.html" height="600"></iframe>
