# In Spain:
# admin_level = 4 means Comunidad Autonoma
# admin_level = 6 means Provincia
# admin_level = 8 means Municipio


# Overpass query to obtain Galicia boundary
[out:json][timeout:25];
{{geocodeArea:Galicia}}->.searchArea;
(
  relation["boundary"="administrative"]["admin_level"="4"](area.searchArea);
);
out body;
>;
out skel qt;


# Overpass query to obtain A Coruña boundary
[out:json][timeout:25];
{{geocodeArea:Coruña}}->.searchArea;
(
  relation["boundary"="administrative"]["admin_level"="6"](area.searchArea);
);
out body;
>;
out skel qt;


# Overpass query to obtain Pontevedra boundary
[out:json][timeout:25];
{{geocodeArea:Pontevedra}}->.searchArea;
(
  relation["boundary"="administrative"]["admin_level"="6"](area.searchArea);
);
out body;
>;
out skel qt;


# Overpass query to obtain Lugo boundary
[out:json][timeout:25];
{{geocodeArea:Lugo}}->.searchArea;
(
  relation["boundary"="administrative"]["admin_level"="6"](area.searchArea);
);
out body;
>;
out skel qt;


# Overpass query to obtain Ourense boundary
[out:json][timeout:25];
{{geocodeArea:Lugo}}->.searchArea;
(
  relation["boundary"="administrative"]["admin_level"="6"](area.searchArea);
);
out body;
>;
out skel qt;

