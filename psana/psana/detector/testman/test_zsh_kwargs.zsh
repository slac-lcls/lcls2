#!/usr/bin/env zsh

# Initialize variables with default values
firstname="First"
secondname="& Second"
thierdname="& Thrierd"

# Loop through all arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -f|--firstname)
            firstname="$2"
            shift # Skip the value argument
            ;;
        -s|--secondname)
            secondname="$2"
            shift # Skip the value argument
            ;;
        -t|--thierdname)
            thierdname="$2"
            shift # Skip the value argument
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # Skip the option argument
done

s="My name is $firstname $secondname $thierdname"
echo $s
#echo "My name is $firstname $secondname $thierdname"
