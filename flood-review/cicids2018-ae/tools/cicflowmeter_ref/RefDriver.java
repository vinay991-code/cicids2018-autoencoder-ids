import cic.cs.unb.ca.jnetpcap.*;
import cic.cs.unb.ca.jnetpcap.worker.FlowGenListener;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Reads a classic pcap (Ethernet, IPv4) and feeds the real CICFlowMeter 2018 FlowGenerator, filling BasicPacketInfo
 *  the way PacketReader.getIpv4Info does (jnetpcap itself is replaced by this reader). Prints one line per flow:
 *  startMicros|srcIP|srcPort|dstIP|dstPort|proto|<dumpFlowBasedFeatures csv> */
public class RefDriver {
    static PrintStream out;
    public static void main(String[] a) throws Exception {
        byte[] d = Files.readAllBytes(Paths.get(a[0]));
        out = new PrintStream(new FileOutputStream(a[1]), false, "UTF-8");
        ByteBuffer b = ByteBuffer.wrap(d);
        int magic = b.order(ByteOrder.LITTLE_ENDIAN).getInt(0);
        boolean nano = false; ByteOrder bo = ByteOrder.LITTLE_ENDIAN;
        if (magic == 0xa1b2c3d4) {} else if (magic == 0xa1b23c4d) nano = true;
        else if (magic == 0xd4c3b2a1) bo = ByteOrder.BIG_ENDIAN; else throw new RuntimeException("not a classic pcap (convert pcapng with editcap -F pcap)");
        b.order(bo);
        int link = b.getInt(20);
        if (link != 1) throw new RuntimeException("only Ethernet captures are handled, linktype " + link);
        FlowGenerator gen = new FlowGenerator(true, 120000000L, 5000000L);
        gen.addFlowListener(new FlowGenListener() { public void onFlowGenerated(BasicFlow f) { emit(f); } });
        IdGenerator ids = new IdGenerator();
        int pos = 24, n = 0, used = 0;
        while (pos + 16 <= d.length) {
            long sec = b.getInt(pos) & 0xffffffffL, frac = b.getInt(pos + 4) & 0xffffffffL;
            int caplen = b.getInt(pos + 8); pos += 16;
            int e = pos; pos += caplen; n++;
            if (caplen < 34) continue;
            int type = ((d[e + 12] & 255) << 8) | (d[e + 13] & 255), l3 = e + 14;
            if (type == 0x8100 && caplen >= 38) { type = ((d[e + 16] & 255) << 8) | (d[e + 17] & 255); l3 = e + 18; }
            if (type != 0x0800) continue;
            int ihl = (d[l3] & 15) * 4, proto = d[l3 + 9] & 255, tot = ((d[l3 + 2] & 255) << 8) | (d[l3 + 3] & 255);
            int fragoff = (((d[l3 + 6] & 31) << 8) | (d[l3 + 7] & 255));
            long ts = sec * 1000000L + (nano ? frac / 1000L : frac);
            BasicPacketInfo p = new BasicPacketInfo(ids);
            p.setSrc(Arrays.copyOfRange(d, l3 + 12, l3 + 16)); p.setDst(Arrays.copyOfRange(d, l3 + 16, l3 + 20));
            p.setTimeStamp(ts);
            int l4 = l3 + ihl;
            if (proto == 6 && fragoff == 0 && l4 + 20 <= d.length) {
                int th = ((d[l4 + 12] & 0xf0) >> 4) * 4, fl = d[l4 + 13] & 255;
                p.setTCPWindow(((d[l4 + 14] & 255) << 8) | (d[l4 + 15] & 255));
                p.setSrcPort(((d[l4] & 255) << 8) | (d[l4 + 1] & 255)); p.setDstPort(((d[l4 + 2] & 255) << 8) | (d[l4 + 3] & 255));
                p.setProtocol(6);
                p.setFlagFIN((fl & 1) != 0); p.setFlagSYN((fl & 2) != 0); p.setFlagRST((fl & 4) != 0); p.setFlagPSH((fl & 8) != 0);
                p.setFlagACK((fl & 16) != 0); p.setFlagURG((fl & 32) != 0); p.setFlagECE((fl & 64) != 0); p.setFlagCWR((fl & 128) != 0);
                p.setPayloadBytes(Math.max(0, tot - ihl - th)); p.setHeaderBytes(th);
            } else if (proto == 17 && fragoff == 0 && l4 + 8 <= d.length) {
                p.setSrcPort(((d[l4] & 255) << 8) | (d[l4 + 1] & 255)); p.setDstPort(((d[l4 + 2] & 255) << 8) | (d[l4 + 3] & 255));
                int ul = ((d[l4 + 4] & 255) << 8) | (d[l4 + 5] & 255);
                p.setPayloadBytes(Math.max(0, ul - 8)); p.setHeaderBytes(8); p.setProtocol(17);
            }   // any other IPv4 packet stays protocol 0, ports 0, payload 0 (as in PacketReader)
            used++;
            gen.addPacket(p);
        }
        Field cf = FlowGenerator.class.getDeclaredField("currentFlows"); cf.setAccessible(true);
        for (Object o : ((Map<?, ?>) cf.get(gen)).values()) { BasicFlow f = (BasicFlow) o; if (f.packetCount() > 1) emit(f); }
        out.close(); System.err.println("packets read " + n + ", IPv4 used " + used);
    }
    static void emit(BasicFlow f) {
        out.println(f.getFlowStartTime() + "|" + f.getSrcIP() + "|" + f.getSrcPort() + "|" + f.getDstIP() + "|" + f.getDstPort() + "|" + f.getProtocol() + "|" + f.dumpFlowBasedFeatures());
    }
}
