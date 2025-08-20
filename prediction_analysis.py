import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_model_and_scaler():
    """
    Load the saved model and scaler
    """
    try:
        model = joblib.load(r'E:\Code alpha Internship\Sales Prediction using python\best_model.pkl')
        scaler = joblib.load(r'E:\Code alpha Internship\Sales Prediction using python\scaler.pkl')
        print("Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def predict_sales(model, scaler, tv_spend, radio_spend, newspaper_spend):
    """
    Predict sales based on advertising spend
    """
    input_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return prediction

def get_user_input():
    """
    Get advertising spend values from user input
    """
    print("\n" + "="*50)
    print("SALES PREDICTION BASED ON ADVERTISING SPEND")
    print("="*50)
    
    while True:
        try:
            tv_spend = float(input("Enter TV advertising spend ($): "))
            if tv_spend < 0:
                print("Please enter a positive value.")
                continue
                
            radio_spend = float(input("Enter Radio advertising spend ($): "))
            if radio_spend < 0:
                print("Please enter a positive value.")
                continue
                
            newspaper_spend = float(input("Enter Newspaper advertising spend ($): "))
            if newspaper_spend < 0:
                print("Please enter a positive value.")
                continue
                
            return tv_spend, radio_spend, newspaper_spend
            
        except ValueError:
            print("Please enter valid numerical values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

def analyze_advertising_impact(model, scaler, tv_spend, radio_spend, newspaper_spend):
    """
    Analyze the impact of advertising on sales for user input
    """
    base_values = np.array([[tv_spend, radio_spend, newspaper_spend]])
    
    base_scaled = scaler.transform(base_values)
    base_pred = model.predict(base_scaled)[0]
    
    impacts = {}
    
    # Test impact of increasing each channel by 10%
    for i, feature in enumerate(['TV', 'Radio', 'Newspaper']):
        test_values = base_values.copy()
        test_values[0, i] += test_values[0, i] * 0.1  # 10% increase
        test_scaled = scaler.transform(test_values)
        test_pred = model.predict(test_scaled)[0]
        
        impact = test_pred - base_pred
        impacts[feature] = impact
    
    return impacts, base_pred

def get_optimization_suggestions(model, scaler, tv_spend, radio_spend, newspaper_spend, base_prediction):
    """
    Get specific suggestions for optimizing advertising spend
    """
    suggestions = []
    
    # Test different optimization strategies
    strategies = [
        {"TV": tv_spend * 1.2, "Radio": radio_spend, "Newspaper": newspaper_spend, "desc": "Increase TV by 20%"},
        {"TV": tv_spend, "Radio": radio_spend * 1.2, "Newspaper": newspaper_spend, "desc": "Increase Radio by 20%"},
        {"TV": tv_spend, "Radio": radio_spend, "Newspaper": newspaper_spend * 1.2, "desc": "Increase Newspaper by 20%"},
        {"TV": tv_spend * 1.1, "Radio": radio_spend * 1.1, "Newspaper": newspaper_spend, "desc": "Increase TV & Radio by 10%"},
        {"TV": tv_spend * 0.9, "Radio": radio_spend * 1.3, "Newspaper": newspaper_spend, "desc": "Reduce TV by 10%, Increase Radio by 30%"},
        {"TV": tv_spend * 1.3, "Radio": radio_spend * 0.9, "Newspaper": newspaper_spend, "desc": "Increase TV by 30%, Reduce Radio by 10%"},
    ]
    
    for strategy in strategies:
        pred = predict_sales(model, scaler, strategy["TV"], strategy["Radio"], strategy["Newspaper"])
        increase = pred - base_prediction
        additional_spend = (strategy["TV"] + strategy["Radio"] + strategy["Newspaper"]) - (tv_spend + radio_spend + newspaper_spend)
        roi = (increase / additional_spend) * 100 if additional_spend > 0 else 0
        
        suggestions.append({
            "description": strategy["desc"],
            "predicted_sales": pred,
            "sales_increase": increase,
            "additional_spend": additional_spend,
            "roi": roi
        })
    
    # Sort by ROI descending
    suggestions.sort(key=lambda x: x["roi"], reverse=True)
    
    return suggestions

def sensitivity_analysis(model, scaler, tv_spend, radio_spend, newspaper_spend, feature_index, range_percent=0.5):
    """
    Perform sensitivity analysis for a specific feature based on user input
    """
    base_values = np.array([[tv_spend, radio_spend, newspaper_spend]])
    variations = np.linspace(-range_percent, range_percent, 11)
    results = []
    
    for variation in variations:
        test_values = base_values.copy()
        test_values[0, feature_index] *= (1 + variation)
        test_scaled = scaler.transform(test_values)
        pred = model.predict(test_scaled)[0]
        results.append((variation * 100, pred))
    
    return results

def generate_comparison_scenarios(model, scaler, tv_spend, radio_spend, newspaper_spend):
    """
    Generate comparison scenarios based on user input
    """
    total_spend = tv_spend + radio_spend + newspaper_spend
    
    scenarios = [
        {
            "TV": tv_spend, 
            "Radio": radio_spend, 
            "Newspaper": newspaper_spend, 
            "Description": "Your Input"
        },
        {
            "TV": tv_spend * 1.2, 
            "Radio": radio_spend, 
            "Newspaper": newspaper_spend, 
            "Description": "20% More TV"
        },
        {
            "TV": tv_spend, 
            "Radio": radio_spend * 1.2, 
            "Newspaper": newspaper_spend, 
            "Description": "20% More Radio"
        },
        {
            "TV": tv_spend, 
            "Radio": radio_spend, 
            "Newspaper": newspaper_spend * 1.2, 
            "Description": "20% More Newspaper"
        },
        {
            "TV": tv_spend * 0.8, 
            "Radio": radio_spend * 1.2, 
            "Newspaper": newspaper_spend, 
            "Description": "Optimized Mix"
        }
    ]
    
    print("\n" + "="*60)
    print("COMPARISON OF DIFFERENT ADVERTISING STRATEGIES")
    print("="*60)
    
    results = []
    for scenario in scenarios:
        prediction = predict_sales(
            model, scaler, 
            scenario["TV"], scenario["Radio"], scenario["Newspaper"]
        )
        scenario_total_spend = scenario["TV"] + scenario["Radio"] + scenario["Newspaper"]
        roi = (prediction - scenario_total_spend) / scenario_total_spend * 100
        
        results.append({
            "Description": scenario["Description"],
            "TV": scenario["TV"],
            "Radio": scenario["Radio"],
            "Newspaper": scenario["Newspaper"],
            "Total Spend": scenario_total_spend,
            "Predicted Sales": prediction,
            "ROI": roi
        })
        
        print(f"\n{scenario['Description']}:")
        print(f"  TV: ${scenario['TV']:.2f}, Radio: ${scenario['Radio']:.2f}, Newspaper: ${scenario['Newspaper']:.2f}")
        print(f"  Total Spend: ${scenario_total_spend:.2f}")
        print(f"  Predicted Sales: ${prediction:.2f}")
        print(f"  Estimated ROI: {roi:.1f}%")
    
    return results

def visualize_results(user_prediction, comparison_results, tv_spend, radio_spend, newspaper_spend, suggestions):
    """
    Visualize the prediction results with optimization suggestions
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Sales Prediction Analysis with Optimization Suggestions', fontsize=16, fontweight='bold')
    
    # 1. Pie chart of current spending
    labels = ['TV', 'Radio', 'Newspaper']
    sizes = [tv_spend, radio_spend, newspaper_spend]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Current Advertising Budget Allocation')
    
    # 2. Bar chart of predicted sales for different scenarios
    scenario_names = [result['Description'] for result in comparison_results]
    sales_values = [result['Predicted Sales'] for result in comparison_results]
    
    bars = axes[0, 1].bar(scenario_names, sales_values, color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[0, 1].set_title('Predicted Sales for Different Scenarios')
    axes[0, 1].set_ylabel('Sales ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, sales_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'${value:.2f}', ha='center', va='bottom')
    
    # 3. ROI comparison for top suggestions
    top_suggestions = suggestions[:3]
    suggestion_names = [s['description'] for s in top_suggestions]
    roi_values = [s['roi'] for s in top_suggestions]
    
    bars = axes[1, 0].bar(suggestion_names, roi_values, color=['green', 'lightgreen', 'yellow'])
    axes[1, 0].set_title('ROI of Top Optimization Suggestions')
    axes[1, 0].set_ylabel('ROI (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, roi_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 4. Text summary with specific suggestions
    axes[1, 1].axis('off')
    
    # Create detailed suggestion text
    suggestion_text = "TOP OPTIMIZATION SUGGESTIONS:\n\n"
    for i, suggestion in enumerate(top_suggestions, 1):
        suggestion_text += f"{i}. {suggestion['description']}:\n"
        suggestion_text += f"   Predicted Sales: ${suggestion['predicted_sales']:.2f}\n"
        suggestion_text += f"   Sales Increase: +${suggestion['sales_increase']:.2f}\n"
        suggestion_text += f"   Additional Spend: ${suggestion['additional_spend']:.2f}\n"
        suggestion_text += f"   Expected ROI: {suggestion['roi']:.1f}%\n\n"
    
    summary_text = f"""
    PREDICTION SUMMARY:
    
    Your Advertising Input:
    - TV: ${tv_spend:.2f}
    - Radio: ${radio_spend:.2f}
    - Newspaper: ${newspaper_spend:.2f}
    - Total Spend: ${tv_spend + radio_spend + newspaper_spend:.2f}
    
    Predicted Sales: ${user_prediction:.2f}
    
    {suggestion_text}
    
    RECOMMENDATION:
    Based on our analysis, {top_suggestions[0]['description']} 
    would generate the highest ROI of {top_suggestions[0]['roi']:.1f}%.
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('sales_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the sales prediction with user input
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return
    
    while True:
        # Get user input
        tv_spend, radio_spend, newspaper_spend = get_user_input()
        
        # Predict sales
        prediction = predict_sales(model, scaler, tv_spend, radio_spend, newspaper_spend)
        
        print("\n" + "="*50)
        print(f"PREDICTED SALES: ${prediction:.2f}")
        print("="*50)
        
        # Analyze impact of advertising
        impacts, base_sales = analyze_advertising_impact(model, scaler, tv_spend, radio_spend, newspaper_spend)
        
        print("\nImpact of 10% Increase in Advertising Spend:")
        for channel, impact in impacts.items():
            print(f"{channel}: +{impact:.2f} sales ({(impact/base_sales)*100:.1f}% increase)")
        
        # Get optimization suggestions
        suggestions = get_optimization_suggestions(model, scaler, tv_spend, radio_spend, newspaper_spend, prediction)
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUGGESTIONS TO INCREASE SALES")
        print("="*60)
        
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"\n{i}. {suggestion['description']}:")
            print(f"   Predicted Sales: ${suggestion['predicted_sales']:.2f}")
            print(f"   Sales Increase: +${suggestion['sales_increase']:.2f}")
            print(f"   Additional Spend: ${suggestion['additional_spend']:.2f}")
            print(f"   Expected ROI: {suggestion['roi']:.1f}%")
        
        # Generate comparison scenarios
        comparison_results = generate_comparison_scenarios(model, scaler, tv_spend, radio_spend, newspaper_spend)
        
        # Sensitivity analysis for TV
        tv_sensitivity = sensitivity_analysis(model, scaler, tv_spend, radio_spend, newspaper_spend, 0)
        print(f"\nTV Advertising Sensitivity Analysis (Base: TV=${tv_spend:.0f}):")
        for change, sales in tv_sensitivity:
            print(f"{change:+.1f}% TV spend: ${sales:.2f} sales")
        
        # Visualize results
        visualize_results(prediction, comparison_results, tv_spend, radio_spend, newspaper_spend, suggestions)
        
        # Ask if user wants to try again
        try_again = input("\nWould you like to try another prediction? (y/n): ").lower()
        if try_again != 'y':
            print("Thank you for using the Sales Prediction Tool!")
            break

if __name__ == "__main__":
    main()