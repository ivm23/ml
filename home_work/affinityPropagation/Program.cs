using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AffinityPropagation
{
    class Info
    {
        public Int32 vertexEnd { get; set; }

        public Double A { get; set; } = 0;
        public Double R { get; set; } = 0;
        public Double S { get; set; }

        public Info(Int32 v2, Int32 ss)
        {
            vertexEnd = v2;
            S = ss;
        }
    }

    struct MaxPair
    {
        public Double FirstMax { get; set; }
        public Double SecondMax { get; set; }
    }

    class Program
    {
        const Int32 N = 196591;
        static Info[][] s_edges = new Info[N][];

        static Int32[] clasters_data = new int[N];

        static void ReadEdges()
        {
            using (StreamReader sr = new StreamReader(@"C:\Users\ivm\source\repos\AffinityPropagation\Gowalla_edges_modify.txt", System.Text.Encoding.Default))
            {
                string line;

                while ((line = sr.ReadLine()) != null)
                {
                    String[] vertex = line.Split(' ');
                    Int32 len = Convert.ToInt32(vertex[1]) + 1;
                    Int32 v = Convert.ToInt32(vertex[0]);
                    s_edges[v] = new Info[len];
                    for (int i = 2; i < len + 1; ++i)
                    {
                        s_edges[v][i - 2] = new AffinityPropagation.Info(Convert.ToInt32(vertex[i]), 1);
                    }
                }
            }
        }

        static void AddKK()
        {
            for (int i = 0; i < N; ++i)
            {
                s_edges[i][s_edges[i].Count() - 1] = new Info(i, -1);
            }
        }

        //static void makeNewDataFormat()
        //{
        //    using (StreamWriter sw = new StreamWriter(@"C:\Users\Marina\Documents\Visual Studio 2015\Projects\AffinityPropagation\AffinityPropagation\Gowalla_edges_modify.txt", true, System.Text.Encoding.Default)) { 

        //        foreach (var a in s_edges)
        //        {
        //            sw.Write(a.Key + " " + a.Value.Count);
        //            foreach(var b in a.Value)
        //            {
        //                sw.Write(" " + b.vertexEnd);
        //            }
        //            sw.WriteLine();
        //        }
        //    }
        //}
        public static Int32 indexFirstMax = -1;
        public static bool fl = false;
        static MaxPair FindMax(Int32 v1) // v1, v2 -- vertex
        {
            Double sumMaxFirst = Double.MinValue;
            Double sumMaxSecond = Double.MinValue;

            for (int k = 0; k < s_edges[v1].Count(); ++k)
            {
                if (k == indexFirstMax)
                    continue;
                if (sumMaxFirst < s_edges[v1][k].A + s_edges[v1][k].S)
                {
                    sumMaxSecond = sumMaxFirst;
                    sumMaxFirst = s_edges[v1][k].A + s_edges[v1][k].S;
                }
            }
            return new MaxPair() { FirstMax = sumMaxFirst, SecondMax = sumMaxSecond };
        }

        static Info KK(Int32 k)
        {
            for (int i = 0; i < s_edges[k].Count(); ++i)
            {
                if (s_edges[k][i].vertexEnd == k)
                    return s_edges[k][i];
            }
            return null;
        }


        static Int32 ArgMax(Int32 v1)
        {
            Int32 index = 0;
            Double max = Double.MinValue;
            foreach (Info k in s_edges[v1])
            {
                Info edge = s_edges[v1].Where(p => p.vertexEnd == k.vertexEnd).First();
                if (edge.R + edge.A > max)
                {
                    max = edge.R + edge.A;
                    index = k.vertexEnd;
                }
            }

            return index;
        }

        private static void saveToFile()
        {
            string FileName = "answer.txt";
            using (StreamWriter sw = new StreamWriter(FileName, true, System.Text.Encoding.Default))
            {
                for (int i = 0; i < clasters_data.Length; i++)
                    sw.WriteLine(clasters_data[i]);
            }
        }

        static void Main(string[] args)
        {
            ReadEdges();
            //  makeNewDataFormat();
            AddKK();
            Double damping = 0.9;

            for (int iter = 0; iter < 10; ++iter)
            {
                Double[] sumMax = new Double[N];
                for (int i = 0; i < N; ++i)
                {
                    MaxPair maxPair = FindMax(i);
                    Double firstMax = maxPair.FirstMax;
                    Double secondMax = maxPair.SecondMax;

                    for (int k = 0; k < s_edges[i].Count(); ++k)
                    {

                        if (k == indexFirstMax)
                        {
                            s_edges[i][k].R = s_edges[i][k].R * damping + (1 - damping) * (s_edges[i][k].S - secondMax);
                        }
                        else
                            s_edges[i][k].R = s_edges[i][k].R * damping + (1 - damping) * (s_edges[i][k].S - firstMax);

                        sumMax[s_edges[i][k].vertexEnd] += Math.Max(0, s_edges[i][k].R);
                    }

                    for (int k = 0; k < s_edges[i].Count(); ++k)
                    {
                        Int32 kv = s_edges[i][k].vertexEnd;
                        Double m1 = Math.Max(0, s_edges[i][k].R);
                        Double m2 = (s_edges[kv].Length > k ? Math.Max(0, s_edges[kv][k].R) : 0);

                        if (i != s_edges[i][k].vertexEnd)
                        {
                            s_edges[i][k].A = s_edges[i][k].R * damping + (1 - damping) * (Math.Min(0, KK(k).R + sumMax[kv] - m1 - m2));
                        }
                        else
                        {
                            s_edges[i][k].A = s_edges[i][k].R * damping + (1 - damping) * (sumMax[kv] - m1 - m2);
                        }
                    }
                }
            }

            for (Int32 i = 0; i < N; ++i)
            {
                clasters_data[i] = ArgMax(i);
            }
            saveToFile();
        }

    }
}
