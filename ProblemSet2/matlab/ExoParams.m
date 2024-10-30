classdef ExoParams
    properties
        beta % discount factor
        delta % depreciation rate
        alpha % capital share
        A % technology parameter
        sigma % standard deviation of income shock
        rho % persistence of income shock
        Nz % number of grid points for income distribution
        Na % number of grid points for asset
        a_lower % lower bound for asset grid
        a_upper % upper bound for asset grid
        m % number of standard deviations for the grid of z
    end

    methods
        function obj = ExoParams()
            obj.beta = 0.96;
            obj.delta = 0.08;
            obj.alpha = 0.36;
            obj.A = 1.0;
            obj.sigma = 0.2;
            obj.rho = 0.9;
            obj.Nz = 7;
            obj.Na = 200;
            obj.a_lower = 0.0;
            obj.a_upper = 150.0;
            obj.m = 3.0;
        end
    end
end
