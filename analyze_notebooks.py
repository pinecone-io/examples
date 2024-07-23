import os
import json
import nbformat
import pandas as pd
from collections import defaultdict

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    is_serverless = False
    clouds = set()
    non_serverless_line = None
    cloud_lines = {}
    creates_pinecone_index = False
    cloud_var = None
    
    for cell_num, cell in enumerate(nb.cells, 1):
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            for line_num, line in enumerate(lines, 1):
                if 'Pinecone(' in line or 'pinecone.create_index(' in line:
                    creates_pinecone_index = True
                    if not is_serverless:
                        non_serverless_line = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                if 'ServerlessSpec' in line:
                    is_serverless = True
                    non_serverless_line = None  # Reset if ServerlessSpec is found
                elif 'Pinecone(' in line and not is_serverless:
                    non_serverless_line = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                
                if 'cloud' in line and '=' in line:
                    cloud_var = line.split('=')[1].strip().strip("'\"")
                    if cloud_var in ['aws', 'gcp', 'azure']:
                        clouds.add(cloud_var)
                        cloud_lines[cloud_var] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                    elif 'os.environ.get' in cloud_var:
                        clouds.add('all-clouds')
                        cloud_lines['all-clouds'] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                
                if 'ServerlessSpec' in line or 'PodSpec' in line:
                    if 'cloud=' in line:
                        cloud = line.split('cloud=')[1].split(',')[0].strip().strip("'\"")
                        if cloud == cloud_var:
                            clouds.add(cloud)
                            cloud_lines[cloud] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                        elif cloud == 'cloud':
                            clouds.add('all-clouds')
                            cloud_lines['all-clouds'] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                if 'environment=' in line:
                    env = line.split('environment=')[1].split(',')[0].strip("'\"")
                    if 'gcp' in env:
                        clouds.add('gcp')
                        cloud_lines['gcp'] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                    elif 'aws' in env:
                        clouds.add('aws')
                        cloud_lines['aws'] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
                    elif 'azure' in env:
                        clouds.add('azure')
                        cloud_lines['azure'] = f"Cell {cell_num}, Line {line_num}: {line.strip()}"
    
    if not creates_pinecone_index:
        return None  # Return None if the notebook doesn't create a Pinecone index
    
    print(f"Analyzed {file_path}: is_serverless={is_serverless}, clouds={clouds}")
    return is_serverless, clouds, non_serverless_line, cloud_lines

def main():
    learn_dir = 'learn'
    results = []
    
    if not os.path.exists(learn_dir):
        print(f"Error: The directory '{learn_dir}' does not exist.")
        print("Current working directory:", os.getcwd())
        return

    print(f"Searching for .ipynb files in '{learn_dir}' and its subdirectories:")
    for root, dirs, files in os.walk(learn_dir):
        for file in files:
            if file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                print(f"Found notebook: {file_path}")
                analysis_result = analyze_notebook(file_path)
                if analysis_result:
                    is_serverless, clouds, non_serverless_line, cloud_lines = analysis_result
                    results.append({
                        'notebook': file_path,
                        'is_serverless': is_serverless,
                        'clouds': ', '.join(clouds) if clouds else 'None specified',
                        'non_serverless_line': non_serverless_line if not is_serverless else 'N/A',
                        'cloud_lines': ', '.join([f"{cloud}: {line}" for cloud, line in cloud_lines.items()])
                    })
                else:
                    print(f"Skipped {file_path}: Does not create a Pinecone index")

    if not results:
        print(f"\nNo .ipynb files found in the '{learn_dir}' directory or its subdirectories.")
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(learn_dir):
            level = root.replace(learn_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        return

    print(f"\nTotal notebooks found: {len(results)}")

    df = pd.DataFrame(results)
    
    # Calculate statistics
    total_notebooks = len(df)
    serverless_notebooks = df['is_serverless'].sum()
    cloud_compatibility = defaultdict(int)
    for clouds in df['clouds']:
        if clouds == 'None specified':
            cloud_compatibility['Unspecified'] += 1
        elif ',' in clouds:
            cloud_compatibility['Multi-cloud'] += 1
        else:
            cloud_compatibility[clouds] += 1
    
    # Print results
    print(f"\nTotal Notebooks: {total_notebooks}")
    print(f"Serverless Notebooks: {serverless_notebooks} ({serverless_notebooks/total_notebooks:.2%})")
    print(f"Non-Serverless Notebooks: {total_notebooks - serverless_notebooks} ({(total_notebooks - serverless_notebooks)/total_notebooks:.2%})")
    print("\nCloud Compatibility:")
    for cloud, count in cloud_compatibility.items():
        print(f"  {cloud}: {count} ({count/total_notebooks:.2%})")
    
    # Save detailed results to CSV
    df.to_csv('notebook_analysis.csv', index=False)
    print("\nDetailed results saved to notebook_analysis.csv")

if __name__ == "__main__":
    main()