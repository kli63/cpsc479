#!/usr/bin/env python3

import os
import json
import random
import argparse
import subprocess
import time
import glob
import threading

lock = threading.Lock()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate style transfer combinations')
    parser.add_argument('--max_combinations', type=int, default=5, 
                        help='Number of combinations to generate in this run')
    parser.add_argument('--progress_file', type=str, default='style_transfer_progress.json',
                        help='File to track progress')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of optimization steps')
    parser.add_argument('--size', type=int, default=512,
                        help='Output image size')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--fresh', action='store_true',
                        help='Generate fresh combinations')
    return parser.parse_args()

def get_image_files(directory):
    return [f for f in glob.glob(os.path.join(directory, "*.jpg")) 
            if not os.path.basename(f).startswith('.')]

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            progress["completed_set"] = set(progress["completed"])
            if "combinations" not in progress:
                progress["combinations"] = {}
            return progress
    return {
        "completed": [],
        "completed_set": set(),
        "in_progress": [],
        "pending": [],
        "combinations": {},
        "total_requested": 0,
        "last_update": time.time()
    }

def save_progress(progress, progress_file):
    with lock:
        progress_copy = progress.copy()
        
        if "completed_set" in progress_copy:
            del progress_copy["completed_set"]
            
        progress_copy["last_update"] = time.time()
        
        with open(progress_file, 'w') as f:
            json.dump(progress_copy, f, indent=2)

def generate_combinations(content_images, style_images, max_combinations, completed_set=None):
    all_combinations = []
    for content in content_images:
        for style in style_images:
            content_id = os.path.splitext(os.path.basename(content))[0]
            style_id = os.path.splitext(os.path.basename(style))[0]
            key = f"{content_id}_{style_id}"
            
            if completed_set and key in completed_set:
                continue
                
            all_combinations.append({
                "content": content,
                "style": style,
                "content_id": content_id,
                "style_id": style_id,
                "key": key
            })
    
    random.shuffle(all_combinations)
    
    return all_combinations[:max_combinations]

def process_combination(combination, args):
    content_path = combination["content"]
    style_path = combination["style"]
    content_id = combination["content_id"]
    style_id = combination["style_id"]
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "style_transfer.py")
    cmd = [
        "python", script_path,
        "--content", content_path,
        "--style", style_path,
        "--steps", str(args.steps),
        "--size", str(args.size)
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    
    print(f"Starting transfer: {content_id} + {style_id}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        return_code = subprocess.call(cmd)
        
        if return_code == 0:
            print(f"Completed transfer: {content_id} + {style_id}")
            return True, combination
        else:
            print(f"Error processing {content_id} + {style_id}: Return code {return_code}")
            return False, combination
            
    except Exception as e:
        print(f"Error processing {content_id} + {style_id}: {e}")
        return False, combination


def main():
    args = parse_arguments()
    
    # Identify our root directory
    model_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(model_dir)
    print(f"Working directory: {os.getcwd()}")
    
    progress_file = os.path.join(model_dir, args.progress_file)
    
    # Limit number of combinations to process
    max_to_process = args.max_combinations
    print(f"Will generate up to {max_to_process} combinations in this run")
    
    progress = load_progress(progress_file)
    print(f"Progress file: {progress_file}")
    
    # Get content and style images
    content_dir = os.path.join(model_dir, "assets", "input")
    style_dir = os.path.join(model_dir, "assets", "reference")
    
    print(f"Content directory: {content_dir}")
    print(f"Style directory: {style_dir}")
    
    content_images = get_image_files(content_dir)
    style_images = get_image_files(style_dir)
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    print(f"Total possible combinations: {len(content_images) * len(style_images)}")
    print(f"Completed combinations: {len(progress['completed'])}")
    
    new_combinations = []
    if args.fresh or len(progress["pending"]) < args.max_combinations:
        need_combinations = args.max_combinations
        if not args.fresh and progress["pending"]:
            need_combinations = need_combinations - len(progress["pending"])
            
        if need_combinations > 0:
            new_combinations = generate_combinations(content_images, style_images, 
                                               need_combinations, 
                                               progress["completed_set"])
            print(f"Generated {len(new_combinations)} new combinations")
        
        if args.fresh:
            print(f"Generating fresh combinations, ignoring {len(progress['pending'])} pending tasks")
            progress["pending"] = []
        
        if new_combinations:    
            progress["pending"].extend([c["key"] for c in new_combinations])
            
            for c in new_combinations:
                progress["combinations"][c["key"]] = c
                
            print(f"Total pending: {len(progress['pending'])}")
            save_progress(progress, progress_file)
            print(f"Saved progress to {progress_file}")
    else:
        # Resume existing pending tasks
        print(f"Resuming with {len(progress['pending'])} pending and {len(progress['in_progress'])} in-progress tasks")
        
        if progress["in_progress"]:
            print(f"Moving {len(progress['in_progress'])} in-progress tasks back to pending:")
            for task in progress["in_progress"]:
                print(f"  - {task}")
            progress["pending"].extend(progress["in_progress"])
            progress["in_progress"] = []
            save_progress(progress, progress_file)
            print(f"Updated progress file")
    
    print("Preparing to start processing...")
    
    to_process = min(max_to_process, len(progress["pending"]))
    print(f"Processing {to_process} combinations sequentially")
    
    for idx in range(to_process):
        if not progress["pending"]:
            print("No more pending combinations to process")
            break
            
        key = progress["pending"].pop(0)
        if key not in progress["combinations"]:
            print(f"WARNING: Key {key} not found in combinations dictionary")
            continue
            
        combination = progress["combinations"][key]
        print(f"\n===== Processing combination {idx+1}/{to_process}: {key} =====")
        
        # Mark as in progress
        progress["in_progress"].append(key)
        save_progress(progress, progress_file)
        
        # Process directly
        success, _ = process_combination(combination, args)
        
        # Update progress
        if key in progress["in_progress"]:
            progress["in_progress"].remove(key)
            
        if success:
            progress["completed"].append(key)
            progress["completed_set"].add(key)
            print(f"===== Completed combination {idx+1}/{to_process}: {key} =====")
        else:
            print(f"===== Failed combination {idx+1}/{to_process}: {key} =====")
            
        save_progress(progress, progress_file)
        print(f"Progress saved: {len(progress['completed'])} combinations completed total")
    
    print(f"Processing complete!")
    
    # Update the gallery manifest
    subprocess.run([os.path.join(model_dir, "..", "gallery", "update-gallery-manifest.sh")], check=True)
    print("Gallery manifest updated.")

if __name__ == "__main__":
    main()