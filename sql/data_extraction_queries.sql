-- ITSM Dataset Exploration Queries

-- View full dataset
SELECT * FROM project_itsm.dataset_list;

-- Impact distribution
SELECT Impact, COUNT(*) 
FROM project_itsm.dataset_list
GROUP BY Impact;

-- Urgency distribution
SELECT Urgency, COUNT(*) 
FROM project_itsm.dataset_list
GROUP BY Urgency;

-- Priority distribution
SELECT Priority, COUNT(*) 
FROM project_itsm.dataset_list
GROUP BY Priority;

-- Total number of records
SELECT COUNT(*) AS total_records
FROM project_itsm.dataset_list;

-- Missing values check
SELECT 
SUM(CASE WHEN Impact IS NULL THEN 1 ELSE 0 END) AS Impact_nulls,
SUM(CASE WHEN Urgency IS NULL THEN 1 ELSE 0 END) AS Urgency_nulls
FROM project_itsm.dataset_list;

-- Relationship between Impact, Urgency, and Priority
SELECT Impact, Urgency, Priority, COUNT(*) 
FROM project_itsm.dataset_list
GROUP BY Impact, Urgency, Priority;

-- Average handling time by priority
SELECT Priority, AVG(Handle_Time_hrs)
FROM project_itsm.dataset_list
GROUP BY Priority;

-- Average reassignments by priority
SELECT Priority, AVG(No_of_Reassignments)
FROM project_itsm.dataset_list
GROUP BY Priority;

-- High priority incidents by configuration category
SELECT CI_Cat, CI_Subcat, COUNT(*) AS high_priority_count
FROM project_itsm.dataset_list
WHERE Priority IN (1,2)
GROUP BY CI_Cat, CI_Subcat;
