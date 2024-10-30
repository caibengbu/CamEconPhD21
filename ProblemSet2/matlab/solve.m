params = ExoParams(); % Exogeneous parameters for aiyagari
[r_tmp, a_stationary_distribution, z_stationary_distribution, joint_transition, v_res, policy_res] = naive_solver(params); % Solve the model using the naive solver
fprintf('capital rent: %.15f \n', r_tmp); % Print the capital rent


N = 1000; % Number of individuals
T = 100; % Number of periods
[a_path_sim, z_path_sim] = simulate_path(joint_transition, N, T, params); % Simulate the path of individuals
plot(a_path_sim', 'HandleVisibility', 'off');  % Plot the simulated path of assets
hold on; % Keep the current plot
plot(a_path_sim(1, :), 'LineWidth', 5, 'DisplayName', 'Individual 1'); % Plot individual 1's path with a thicker line
mean_path = mean(a_path_sim, 1); % Calculate the mean path of assets
plot(mean_path, 'LineWidth', 5, 'DisplayName', 'Mean Wealth');
legend('show'); % Show the legend
hold off; % Release the plot hold
saveas(gcf, 'path_simulation.png'); % Save the plot as a PNG file

% Collect top 10% at t = 50
[~, sorted_indices] = sort(a_path_sim(:, 50));
to_keep = sorted_indices(901:end);
a_path_sim_collect_top = a_path_sim(to_keep, 51:end);
mean_path_collect_top = mean(a_path_sim_collect_top, 1);
% Calculate 95% confidence interval for the mean path after t = 50
mean_path_collect_top_ci = prctile(a_path_sim_collect_top, [2.5, 97.5], 1);
plot(mean_path_collect_top, 'r', 'LineWidth', 5, 'DisplayName', 'Mean Wealth (Collect Top 10% at t = 50)');
hold on;

% Plot the confidence interval
fill([1:50, fliplr(1:50)], [mean_path_collect_top_ci(1, 1:50), fliplr(mean_path_collect_top_ci(2, 1:50))], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '95% CI');
legend('show'); % Update the legend

% Cut bottom 10% at t = 50
to_keep = sorted_indices(1:100);
a_path_sim_collect_btm = a_path_sim(to_keep, 51:end);
mean_path_collect_btm = mean(a_path_sim_collect_btm, 1);

% Calculate 95% confidence interval for the mean path after t = 50
mean_path_collect_btm_ci = prctile(a_path_sim_collect_btm, [2.5, 97.5], 1);
plot(mean_path_collect_btm, 'b', 'LineWidth', 5, 'DisplayName', 'Mean Wealth (Collect Bottom 10% at t = 50)');

% Plot the confidence interval
fill([1:50, fliplr(1:50)], [mean_path_collect_btm_ci(1, 1:50), fliplr(mean_path_collect_btm_ci(2, 1:50))], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '95% CI');
legend('show'); % Update the legend
hold off;
saveas(gcf, 'path_simulation_2.png'); % Save the plot as a PNG file

% plot lorentz curve
wealth_distribution = a_path_sim(:, 51:end);
wealth_distribution = reshape(wealth_distribution, [], 1);
sorted_wealth = sort(wealth_distribution);
cumulative_wealth = cumsum(sorted_wealth);
cumulative_wealth_share = cumulative_wealth / cumulative_wealth(end);
population_share = (1:length(sorted_wealth))' / length(sorted_wealth);

% Calculate Gini coefficient
AUC_lorenz = trapz(population_share, cumulative_wealth_share);
AUC_line_of_equality = trapz([0, 1], [0, 1]);
gini_coefficient = (AUC_line_of_equality - AUC_lorenz) / AUC_line_of_equality;
fprintf('Gini Coefficient: %f\n', gini_coefficient); % Print the Gini coefficient

% Plot the Lorenz curve
figure;
plot(population_share, cumulative_wealth_share, 'LineWidth', 2, 'DisplayName', 'Lorenz Curve');
hold on;

% Plot the line of equality
plot([0, 1], [0, 1], '--k', 'DisplayName', 'Line of Equality');

legend('show'); % Show the legend
xlabel('Cumulative Share of Population');
ylabel('Cumulative Share of Wealth');
title('Lorenz Curve');
text(0.6, 0.2, sprintf('Gini Coefficient: %.6f', gini_coefficient), 'FontSize', 12, 'BackgroundColor', 'white');
hold off;
saveas(gcf, 'lorenz_curve.png'); % Save the plot as a PNG file

% Counterfactural: 20% increase in productivity (A)
params.A = 1.2;
[r_tmp, a_stationary_distribution_2, z_stationary_distribution, joint_transition, v_res, policy_res] = naive_solver(params); % Solve the model with the new productivity level
N = 1000; % Number of individuals
T = 100; % Number of periods
[a_path_sim, z_path_sim] = simulate_path(joint_transition, N, T, params); % Simulate the path of individuals

% plot lorentz curve
wealth_distribution = a_path_sim(:, 51:end);
wealth_distribution = reshape(wealth_distribution, [], 1);
sorted_wealth = sort(wealth_distribution);
cumulative_wealth = cumsum(sorted_wealth);
cumulative_wealth_share = cumulative_wealth / cumulative_wealth(end);
population_share = (1:length(sorted_wealth))' / length(sorted_wealth);

% Calculate Gini coefficient
AUC_lorenz = trapz(population_share, cumulative_wealth_share);
AUC_line_of_equality = trapz([0, 1], [0, 1]);
gini_coefficient = (AUC_line_of_equality - AUC_lorenz) / AUC_line_of_equality;
fprintf('Gini Coefficient: %f\n', gini_coefficient); % Print the Gini coefficient

% Plot the Lorenz curve
figure;
plot(population_share, cumulative_wealth_share, 'LineWidth', 2, 'DisplayName', 'Lorenz Curve');
hold on;

% Plot the line of equality
plot([0, 1], [0, 1], '--k', 'DisplayName', 'Line of Equality');

legend('show'); % Show the legend
xlabel('Cumulative Share of Population');
ylabel('Cumulative Share of Wealth');
title('Lorenz Curve');
text(0.6, 0.2, sprintf('Gini Coefficient: %.6f', gini_coefficient), 'FontSize', 12, 'BackgroundColor', 'white');
hold off;
saveas(gcf, 'lorenz_curve_counterfactual.png'); % Save the plot as a PNG file


a_grid = get_a_grid(params);
a_sample = randsample(1:params.Na, 100000, true, a_stationary_distribution);
a_sample_2 = randsample(1:params.Na, 100000, true, a_stationary_distribution_2);
a_sample = a_grid(a_sample);
a_sample_2 = a_grid(a_sample_2);
relative_variation = std(a_sample) / mean(a_sample);
relative_variation_2 = std(a_sample_2) / mean(a_sample_2);
fprintf('When A = 1, std(wealth)/mean(wealth) = %.6f\n', relative_variation);
fprintf('When A = 1.2, std(wealth)/mean(wealth) = %.6f\n', relative_variation_2);






