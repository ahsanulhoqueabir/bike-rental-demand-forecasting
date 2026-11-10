# Task 8 – Train vs validation error and predictions over validation dates

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Compare train vs validation error and plot predictions over validation dates; interpret and acknowledge uncertainty.

## 2. Why this approach
Diagnostics tell whether the model over-fits or the data shifts – both inflate validation error but call for different remedies.

## 3. How it was implemented
Refit the selected forest; table of train/validation MAE, RMSE, R², RMSE ratio, mean bias; time plot of actual vs RF/Linear/Dummy over validation dates with residual bars (Fig. 8); scatter of train vs validation RMSE for all 48 forests (Fig. 9).

## 4. Logic behind the implementation
A residual bar plot over time shows *systematic* error (bias) in contrast to random noise.

## 5. Important decisions, assumptions and considerations
* Train RMSE 470.5 → validation 825.8 (1.76×; default forest 3.7×).
* Mean validation bias −285: peaks are flattened.
* Interpretation therefore separates variance from shift; uncertainty is stated (109 days, one season).

## 6. Expected output and how to read it
Fig. 8 shows the forest follows the weekly up/down pattern but sits below actual demand on most days.
