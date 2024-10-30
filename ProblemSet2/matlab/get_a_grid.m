function a_grid = get_a_grid(params)
    % Extract parameters
    Na = params.Na;
    a_lower = params.a_lower;
    a_upper = params.a_upper;
    
    x = linspace(0,0.5,Na);
    y = x.^5/max(x.^5);
    a_grid = a_lower+(a_upper-a_lower)*y;
end