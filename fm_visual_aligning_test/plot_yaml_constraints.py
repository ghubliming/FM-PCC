import os
import yaml
import sys

# Ensure the current directory is in sys.path so we can import eval_fm_visual_aligning
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval_fm_visual_aligning import plot_geo_constraints

def main():
    # Setup paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/visual_aligning_eval.yaml')
    out_base_dir = os.path.join(script_dir, 'constraint_plots')
    
    print(f"Loading config from: {os.path.abspath(config_path)}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    geo_variants = config.get('geo_constraint_variants', [])
    if not geo_variants:
        print("No geo_constraint_variants found in config.")
        return
        
    print(f"Found {len(geo_variants)} geometric constraint variants.")
    enlarge = config.get('enlarge_constraints', 0.0)
    
    for geo_config in geo_variants:
        geo_name = geo_config.get('name', 'unnamed')
        out_dir = os.path.join(out_base_dir, geo_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Inject global enlarge_constraints into the geo_config dictionary 
        # so that plot_geo_constraints can access it
        if enlarge:
            geo_config['enlarge_constraints'] = enlarge
            
        print(f"Plotting constraints for: {geo_name}")
        out_file = os.path.join(out_dir, 'constraint_overview.png')
        if os.path.exists(out_file):
            os.remove(out_file)
            
        # Plot base configuration
        plot_geo_constraints(geo_name, geo_config, out_dir, is_tightened=False)
        
        # Generate tightened variant if applicable (bounds, obstacles, or halfspace)
        if enlarge and enlarge > 0.0:
            c_types = geo_config.get('constraint_types', [])
            if 'bounds' in c_types or 'obstacles' in c_types or 'halfspace' in c_types:
                geo_name_tightened = f"{geo_name}-tightened"
                out_dir_tightened = os.path.join(out_base_dir, geo_name_tightened)
                os.makedirs(out_dir_tightened, exist_ok=True)
                
                print(f"Plotting constraints for: {geo_name_tightened}")
                out_file_tight = os.path.join(out_dir_tightened, 'constraint_overview.png')
                if os.path.exists(out_file_tight):
                    os.remove(out_file_tight)
                    
                # Plot tightened configuration
                plot_geo_constraints(geo_name_tightened, geo_config, out_dir_tightened, is_tightened=True)

    print(f"\nAll plots saved to: {out_base_dir}")

if __name__ == '__main__':
    main()
