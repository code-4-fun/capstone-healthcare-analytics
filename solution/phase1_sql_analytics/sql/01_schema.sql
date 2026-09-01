-- Phase 1 :: Schema bootstrap
-- Creates the dedicated analytics schema. Idempotent.

CREATE SCHEMA IF NOT EXISTS capstone_solution;

COMMENT ON SCHEMA capstone_solution IS
    'Hospital Operations & Revenue Risk Intelligence Platform - Phase 1 SQL analytics layer';

SET search_path TO capstone_solution, public;
