# Understanding L2 Error in Convergence Tests

## What is L2 Error?

The **L2 error** (also called **Root Mean Square Error** or **RMS error**) measures how much a test solution deviates from a reference solution. It's calculated as:

```
L2_error = √(mean((reference - test)²))
```

This gives you a single number that represents the "average" magnitude of differences across all time steps and spatial locations.

## In Your Convergence Test

### Reference Solution
- **Reference**: `dt_safety_factor = 0.45` (finest time step, most accurate)
- All other solutions are compared against this reference

### What Each L2 Error Means

#### 1. **h_total L2 Error** (Total Thickness Error)
- **Units**: meters (m)
- **What it measures**: How much the total frost thickness differs from the reference
- **Your results**:
  - `sf=0.50`: `8.11e-07 m` = **0.0008 μm** (excellent agreement!)
  - `sf=0.90`: `1.39e-04 m` = **139 μm** (larger difference)

**Interpretation**: 
- Values < 1 μm (1e-6 m) are excellent
- Values < 10 μm (1e-5 m) are very good
- Values > 100 μm (1e-4 m) show noticeable differences

#### 2. **Temperature L2 Error**
- **Units**: degrees Celsius (°C)
- **What it measures**: How much temperatures differ across all layers and time steps
- **Your results**:
  - `sf=0.50`: `3.77e-04 °C` = **0.0004°C** (excellent!)
  - `sf=0.90`: `8.70e-01 °C` = **0.87°C** (noticeable difference)

**Interpretation**:
- Values < 0.01°C are excellent
- Values < 0.1°C are very good
- Values > 0.5°C show meaningful differences

#### 3. **Alpha_ice L2 Error** (Ice Volume Fraction Error)
- **Units**: dimensionless (fraction between 0 and 1)
- **What it measures**: How much the ice volume fraction differs
- **Your results**:
  - `sf=0.50`: `4.88e-03` = **0.49%** (very good)
  - `sf=0.90`: `1.04e-01` = **10.4%** (significant difference)

**Interpretation**:
- Values < 0.01 (1%) are excellent
- Values < 0.05 (5%) are good
- Values > 0.10 (10%) show substantial differences

## Understanding Your Results

### Excellent Convergence (sf = 0.50)
```
h_total L2:    8.11e-07 m    (0.0008 μm)  ← Excellent!
Temperature:   3.77e-04 °C   (0.0004°C)   ← Excellent!
Alpha_ice:     4.88e-03      (0.49%)      ← Very Good!
```

This means using `dt_safety_factor = 0.50` gives results that are virtually identical to the reference solution.

### Good Convergence (sf = 0.60)
```
h_total L2:    1.43e-05 m    (14.3 μm)    ← Very Good
Temperature:   5.04e-02 °C   (0.05°C)     ← Very Good
Alpha_ice:     5.59e-02      (5.6%)       ← Good
```

Still very accurate, but with slightly larger differences.

### Noticeable Differences (sf = 0.90)
```
h_total L2:    1.39e-04 m    (139 μm)     ← Noticeable
Temperature:   8.70e-01 °C   (0.87°C)     ← Noticeable
Alpha_ice:     1.04e-01      (10.4%)      ← Significant
```

The larger time step introduces measurable errors.

## Why L2 Error Matters

1. **Convergence**: As `dt_safety_factor` increases (larger time steps), errors should increase. This confirms your solver is working correctly.

2. **Accuracy vs Speed Trade-off**:
   - **sf = 0.50**: Very accurate, but slower (more time steps)
   - **sf = 0.90**: Faster, but less accurate

3. **Recommendation**: The script recommends `sf = 0.50` because:
   - Errors are extremely small (excellent accuracy)
   - Still faster than the reference (sf = 0.45)
   - Good balance between accuracy and computational cost

## Visual Interpretation

The L2 error is like the "average distance" between two curves:
- **Small L2 error** = curves are very close together
- **Large L2 error** = curves diverge significantly

For your frost defrost simulation:
- **h_total L2 < 1 μm**: Thickness predictions are essentially identical
- **Temperature L2 < 0.01°C**: Temperature predictions are essentially identical
- **Alpha_ice L2 < 1%**: Phase fractions are essentially identical

## Summary

Your convergence test shows that:
- ✅ `dt_safety_factor = 0.50` provides excellent accuracy
- ✅ The solver converges properly (errors increase with larger time steps)
- ✅ You can safely use `sf = 0.50` for production runs with confidence

The L2 errors at `sf = 0.50` are so small they're essentially within numerical precision, meaning you're getting reference-quality results with better computational efficiency!
