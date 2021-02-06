# Ports mapping using behavioral vessel data

## Problem description
Ports are the departure and arrival points of almost all vessel journeys and play a major role in the supply chain of various goods. As such, ports --  and associated  areas within or around them (namely: docks and port waiting areas) -- are a central point of interest in the maritime world. Therefore, being able to identify them, locate them accurately and distinguish between them can serve various needs of Windwards customers. Though along the years Windward had gained lots of information about them, their mapping and coverage are far from being complete.
In this project we will use vessel behavioral data in order to map port related areas of interest.

 

## Dataset description 
Windward will provide various data sets that contain:

Vessel activities - millions of records of relevant behavioral patterns including: port calls, drifting anchoring and mooring. The data contains vessel identifier, geolocation, type of activity, time information and other activity related parameters

Vessel information - vessel identifiers, vessel class, vessel measures, vessel flag, additional technical details of around one million vessels

Layer polygons - geospatial information about port and other areas of interest locations and boundaries
Port information - technical port details
 

## Project goals
Basic goal – Mapping of ports and waiting areas. Create sets of area candidates according to vessels’ behavior, and based on tagged data, create a model that classifies areas as ports, port waiting areas or other.

Advanced goal – Mapping of docks and terminals within ports. This is an unsupervised task (since there is very limited tagged dock data) which will require to distinguish between docks and terminals within ports.
Evaluation – As the real value of the models are for areas which are unknown to Windward, the evaluation will be done using standard metrics (precision, recall, accuracy etc.) w.r.t. samples that will be labeled especially for this purpose. For the advanced goal, our analyst team will review the candidates extracted by the students and evaluate them.

## Project deliverables:
Basic deliverables - geolocations of new ports (i.e. that are not mapped in Windwards databases), geolocations of new port waiting areas, the code and documentation of the methodology used.

Advanced deliverables - geolocations of docks, the code and documentation of the methodology used.

## Project impact for the company/product:
Naturally increasing the coverage is valuable for Windward. Moreover,  the immediate impact is on a new line of cargo monitoring products for which it is highly important to be able to be able to identify each port call and\or dock visit.
