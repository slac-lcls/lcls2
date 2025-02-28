while [ 1 ]
do
  daqstate -p 2 -P rix -C drp-srcf-cmp004 --state running
  sleep 2
  daqstate -p 2 -P rix -C drp-srcf-cmp004 --state paused
  sleep 2
done
