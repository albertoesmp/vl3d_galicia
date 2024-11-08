-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the global evaluation of the building and
--          vegetation experiments. The global evaluation considers all the 
--          predicted point clouds for the building and vegetation 
--          classification task.
-- Brief: SQL query to compute the class-wise recount of the points in the
--          original class system (i.e., before transforming the set of classes
--          to represent a particular experiment).




-- Sort point clouds by class balance in GALICIA
SELECT datasets.name as "Pcloud name",
	min(class_distributions.recount) as "Min class abs-freq.",
	100.0*min(class_distributions.recount)/datasets.num_points as "Min class rel-freq. (%)",
	max(class_distributions.recount) as "Max class abs-freq.",
	100.0*max(class_distributions.recount)/datasets.num_points as "Max class rel-freq. (%)",
	max(class_distributions.recount) - min(class_distributions.recount) as "Maxmin abs-freq. diff.",
	100.0*(
		max(class_distributions.recount) - min(class_distributions.recount)
	)/datasets.num_points as "Maxmin rel-freq. diff. (%)",
	datasets.num_points as "Points recount"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
WHERE datasets.name like '%_ORIGINAL'
	AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
GROUP BY datasets.id
--ORDER BY "Maxmin abs-freq. diff." asc
--ORDER BY "Min class abs-freq." desc
ORDER BY "Min class rel-freq. (%)" desc




-- Sort point clouds by class balance in A CORUÑA
SELECT datasets.name as "Pcloud name",
	geographic_regions.name as "Region",
    ( -- Name of class with min relative frequency
        SELECT classes.name
        FROM class_distributions
            JOIN classes ON class_distributions.class_id = classes.id
        WHERE class_distributions.dataset_id = datasets.id
            AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
        ORDER BY class_distributions.recount asc
        LIMIT 1
    ) AS "Min rel-freq. class",
	min(class_distributions.recount) as "Min class abs-freq.",
	100.0*min(class_distributions.recount)/datasets.num_points as "Min class rel-freq. (%)",
	max(class_distributions.recount) as "Max class abs-freq.",
	100.0*max(class_distributions.recount)/datasets.num_points as "Max class rel-freq. (%)",
	max(class_distributions.recount) - min(class_distributions.recount) as "Maxmin abs-freq. diff.",
	100.0*(
		max(class_distributions.recount) - min(class_distributions.recount)
	)/datasets.num_points as "Maxmin rel-freq. diff. (%)",
	datasets.num_points as "Points recount",
	sum(  -- Count of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'A Coruña'
		) THEN
			dataset_regions.num_points
		END
	) as "Points in region",
	100.0*sum(  -- Percentage of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'A Coruña'
		) THEN
			dataset_regions.num_points
		END
	) / datasets.num_points as "Points in region (%)"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE datasets.name like '%_ORIGINAL'
	AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
	AND geographic_regions.name like 'A Coruña'
GROUP BY datasets.id, geographic_regions.id
HAVING sum(
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'A Coruña'
		) THEN
			dataset_regions.num_points
		END
	)/datasets.num_points >= 0.9
--ORDER BY "Maxmin abs-freq. diff." asc
--ORDER BY "Min class abs-freq." desc
ORDER BY "Min class rel-freq. (%)" desc





-- Sort point clouds by class balance in PONTEVEDRA
SELECT datasets.name as "Pcloud name",
	geographic_regions.name as "Region",
    ( -- Name of class with min relative frequency
        SELECT classes.name
        FROM class_distributions
            JOIN classes ON class_distributions.class_id = classes.id
        WHERE class_distributions.dataset_id = datasets.id
            AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
        ORDER BY class_distributions.recount asc
        LIMIT 1
    ) AS "Min rel-freq. class",
	min(class_distributions.recount) as "Min class abs-freq.",
	100.0*min(class_distributions.recount)/datasets.num_points as "Min class rel-freq. (%)",
	max(class_distributions.recount) as "Max class abs-freq.",
	100.0*max(class_distributions.recount)/datasets.num_points as "Max class rel-freq. (%)",
	max(class_distributions.recount) - min(class_distributions.recount) as "Maxmin abs-freq. diff.",
	100.0*(
		max(class_distributions.recount) - min(class_distributions.recount)
	)/datasets.num_points as "Maxmin rel-freq. diff. (%)",
	datasets.num_points as "Points recount",
	sum(  -- Count of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Pontevedra'
		) THEN
			dataset_regions.num_points
		END
	) as "Points in region",
	100.0*sum(  -- Percentage of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Pontevedra'
		) THEN
			dataset_regions.num_points
		END
	) / datasets.num_points as "Points in region (%)"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE datasets.name like '%_ORIGINAL'
	AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
	AND geographic_regions.name like 'Pontevedra'
GROUP BY datasets.id, geographic_regions.id
HAVING sum(
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Pontevedra'
		) THEN
			dataset_regions.num_points
		END
	)/datasets.num_points >= 0.9
--ORDER BY "Maxmin abs-freq. diff." asc
--ORDER BY "Min class abs-freq." desc
ORDER BY "Min class rel-freq. (%)" desc




-- Sort point clouds by class balance in LUGO
SELECT datasets.name as "Pcloud name",
	geographic_regions.name as "Region",
    ( -- Name of class with min relative frequency
        SELECT classes.name
        FROM class_distributions
            JOIN classes ON class_distributions.class_id = classes.id
        WHERE class_distributions.dataset_id = datasets.id
            AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
        ORDER BY class_distributions.recount asc
        LIMIT 1
    ) AS "Min rel-freq. class",
	min(class_distributions.recount) as "Min class abs-freq.",
	100.0*min(class_distributions.recount)/datasets.num_points as "Min class rel-freq. (%)",
	max(class_distributions.recount) as "Max class abs-freq.",
	100.0*max(class_distributions.recount)/datasets.num_points as "Max class rel-freq. (%)",
	max(class_distributions.recount) - min(class_distributions.recount) as "Maxmin abs-freq. diff.",
	100.0*(
		max(class_distributions.recount) - min(class_distributions.recount)
	)/datasets.num_points as "Maxmin rel-freq. diff. (%)",
	datasets.num_points as "Points recount",
	sum(  -- Count of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Lugo'
		) THEN
			dataset_regions.num_points
		END
	) as "Points in region",
	100.0*sum(  -- Percentage of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Lugo'
		) THEN
			dataset_regions.num_points
		END
	) / datasets.num_points as "Points in region (%)"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE datasets.name like '%_ORIGINAL'
	AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
	AND geographic_regions.name like 'Lugo'
GROUP BY datasets.id, geographic_regions.id
HAVING sum(
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Lugo'
		) THEN
			dataset_regions.num_points
		END
	)/datasets.num_points >= 0.9
--ORDER BY "Maxmin abs-freq. diff." asc
--ORDER BY "Min class abs-freq." desc
ORDER BY "Min class rel-freq. (%)" desc




-- Sort point clouds by class balance in OURENSE
SELECT datasets.name as "Pcloud name",
	geographic_regions.name as "Region",
    ( -- Name of class with min relative frequency
        SELECT classes.name
        FROM class_distributions
            JOIN classes ON class_distributions.class_id = classes.id
        WHERE class_distributions.dataset_id = datasets.id
            AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
        ORDER BY class_distributions.recount asc
        LIMIT 1
    ) AS "Min rel-freq. class",
	min(class_distributions.recount) as "Min class abs-freq.",
	100.0*min(class_distributions.recount)/datasets.num_points as "Min class rel-freq. (%)",
	max(class_distributions.recount) as "Max class abs-freq.",
	100.0*max(class_distributions.recount)/datasets.num_points as "Max class rel-freq. (%)",
	max(class_distributions.recount) - min(class_distributions.recount) as "Maxmin abs-freq. diff.",
	100.0*(
		max(class_distributions.recount) - min(class_distributions.recount)
	)/datasets.num_points as "Maxmin rel-freq. diff. (%)",
	datasets.num_points as "Points recount",
	sum(  -- Count of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Ourense'
		) THEN
			dataset_regions.num_points
		END
	) as "Points in region",
	100.0*sum(  -- Percentage of points in region
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Ourense'
		) THEN
			dataset_regions.num_points
		END
	) / datasets.num_points as "Points in region (%)"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE datasets.name like '%_ORIGINAL'
	AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
	AND geographic_regions.name like 'Ourense'
GROUP BY datasets.id, geographic_regions.id
HAVING sum(
		DISTINCT case when dataset_regions.region_id in (
			SELECT geographic_regions.id
			FROM geographic_regions
			WHERE geographic_regions.name like 'Ourense'
		) THEN
			dataset_regions.num_points
		END
	)/datasets.num_points >= 0.9
--ORDER BY "Maxmin abs-freq. diff." asc
--ORDER BY "Min class abs-freq." desc
ORDER BY "Min class rel-freq. (%)" desc
