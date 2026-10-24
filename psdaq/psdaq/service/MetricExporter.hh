/**
 * @file
 * @brief Prometheus monitoring helpers: exposer creation, a histogram type, and MetricExporter, a collectable that reads application counters on each scrape.
 */
#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <prometheus/exposer.h>
#include <prometheus/metric_type.h>
#include <prometheus/metric_family.h>

namespace Pds
{

/** Same as createExposer(prometheusDir, hostname, 0). */
std::unique_ptr<prometheus::Exposer>
    createExposer(const std::string& prometheusDir,
                  const std::string& hostname);

// Entry for allowing app to attempt to pick the same port on each invocation
/** Start a Prometheus exposer on the first usable port from 9200 + portOffset (wrapping within 9200 to 9299) and write the scrape target file prometheusDir/drpmon_HOST_N.yaml for it. Returns an empty pointer if prometheusDir is empty or no port works. */
std::unique_ptr<prometheus::Exposer>
    createExposer(const std::string& prometheusDir,
                  const std::string& hostname,
                  unsigned           portOffset);

/** Last sample of a Rate metric (value and time), used to compute the rate at the next scrape. */
struct Previous
{
    int64_t value;  ///< Counter value at the previous scrape.
    std::chrono::steady_clock::time_point time;  ///< Time of the previous scrape.
};

/** Fixed-bin histogram exported as a Prometheus histogram: numBins upper bounds binMin + i * binWidth plus an overflow bucket. */
class PromHistogram
{
public:
    /** Upper bounds of the buckets. */
    using BucketBoundaries = std::vector<double>;

    /** Set numBins upper bounds starting at binMin with spacing binWidth, and numBins + 1 zeroed counts (the last is the overflow bucket). */
    PromHistogram(unsigned numBins, double binWidth, double binMin);

    /** Count value in the first bucket whose upper bound is at least value (the overflow bucket if none) and add it to the sum. */
    void observe(double value);
    /** Fill the histogram buckets (cumulative counts, upper bounds with infinity for the last), sample count and sum of the first metric in family. */
    void collect(prometheus::MetricFamily& family);
    /** Zero all counts and the sum. */
    void clear();
    /** Return the count of bucket idx. */
    uint64_t counts(unsigned idx) const { return m_counts[idx]; }
    /** Print the count of every bucket to stdout. */
    void dump() const;

private:
    BucketBoundaries      m_boundaries;
    std::vector<uint64_t> m_counts;
    double                m_sum;
};

/** Kind of a metric registered with MetricExporter. */
enum class MetricType
{
    Gauge,  ///< Value read by the callback at each scrape, exported as a gauge.
    Counter,  ///< Value read by the callback at each scrape, exported as a counter.
    Rate,  ///< Rate per second of the callback value since the previous scrape (negative changes give 0), exported as a gauge.
    /** Fixed value set by constant(), exported as a counter (per the comment, only for constant()). */
    Constant,                           // To be used only by addConst()
    Histogram,  ///< A PromHistogram created by histogram().
    Float  ///< Double value from a callback that reports whether it is valid, exported as a gauge.
};

/** Prometheus collectable holding named metrics whose values come from callbacks evaluated at each scrape (Collect()). Adding a metric with an existing name replaces it. */
class MetricExporter : public prometheus::Collectable
{
public:
    /** Return the index of the metric named name, or -1. */
    int find(const std::string& name) const;
    /** Register metric name with labels, of the given type, whose value is read by calling value; an existing metric with that name is replaced. For Rate, value is called once now to set the starting point. */
    void add(const std::string& name,
             const std::map<std::string, std::string>& labels,
             MetricType type, std::function<int64_t()> value);
    /** Register a Float gauge name whose callback stores a double and returns whether it is valid; an existing metric with that name is replaced. */
    void addFloat(const std::string& name,
                  const std::map<std::string, std::string>& labels,
                  std::function<bool(double&)> value);
    /** Register a Constant metric name exported as a counter with the fixed value value. */
    void constant(const std::string& name,
                  const std::map<std::string, std::string>& labels,
                  int64_t value);
    /** Register a histogram metric name with numBins buckets of width binWidth starting at binMin and return it for observe() calls. Note that, unlike add() and addFloat(), this does not append a Float callback slot, so the internal per-metric vectors can get out of step. */
    std::shared_ptr<PromHistogram>
         histogram(const std::string& name,
                   const std::map<std::string, std::string>& labels,
                   unsigned numBins, double binWidth=1.0, double binMin=0.0);
    /** Update every metric (rates, histograms, valid floats and callback values; constants unchanged) and return the metric families. */
    std::vector<prometheus::MetricFamily> Collect() const override;
private:
    void _erase(unsigned index);
private:
    mutable std::mutex m_mutex;
    mutable std::vector<prometheus::MetricFamily> m_families;
    std::vector<std::function<int64_t()> > m_values;
    std::vector<std::function<bool(double&)> > m_floats;
    mutable std::vector<std::shared_ptr<PromHistogram> > m_histos;
    std::vector<MetricType> m_type;
    mutable std::vector<Previous> m_previous;
};

/** Set the counter or gauge value of the first metric of family to value; prints wrong type for other metric types. */
template<typename T>
void setValue(prometheus::MetricFamily& family, const T& value)
{
    switch (family.type) {
        case prometheus::MetricType::Counter: {
            family.metric[0].counter.value = static_cast<double>(value);
            break;
        }
        case prometheus::MetricType::Gauge: {
            family.metric[0].gauge.value = static_cast<double>(value);
            break;
        }
        default:
            std::cout<<"wrong type\n";
    }
}

}
